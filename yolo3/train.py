from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import neural_structured_learning as nsl
import numpy as np


class AdvLossModel(tf.keras.Model):
    def _compute_total_loss(self, y_trues, y_preds, sample_weights=None):
        loss = 0
        for y_true, y_pred, loss_object in zip(y_trues, y_preds, self.loss):
            loss += loss_object(y_true, y_pred)
        return loss

    def _train_step(self, inputs):
        images, y_trues = inputs
        with tf.GradientTape() as tape_w:
            tape_w.watch(self.trainable_variables)
            if self.use_adv:
                with tf.GradientTape() as tape_x:
                    tape_x.watch(images)
                    y_preds = self(images, training=True)
                    loss = self._compute_total_loss(y_trues, y_preds)
                    adv_loss = nsl.keras.adversarial_loss(
                        images,
                        y_trues,
                        self,
                        self._compute_total_loss,
                        labeled_loss=loss,
                        gradient_tape=tape_x,
                        adv_config=self.adv_config)
                    loss += self.adv_config.multiplier * adv_loss
            else:
                y_preds = self(images, training=True)
                loss = self._compute_total_loss(y_trues, y_preds)
        gradients = tape_w.gradient(loss, self.trainable_variables)
        optimizer_op = self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))
        if self.use_ema:
            ema = tf.train.ExponentialMovingAverage(decay=0.9999,zero_debias=True)
            with tf.control_dependencies([optimizer_op]):
                ema.apply(self.trainable_variables)
        return loss

    def _val_step(self, inputs):
        images, y_trues = inputs
        y_preds = self(images, training=False)
        loss = self._compute_total_loss(y_trues, y_preds)
        return loss

    @tf.function
    def _distributed_epoch(self, dataset, step):
        total_loss = 0.0
        num_batches = 0.0
        for batch in dataset:
            if self.writer is not None:
                with self.writer.as_default():
                    tf.summary.image(
                        "Training data",
                        tf.cast(batch[0] * 255, tf.uint8),
                        max_outputs=8)
            per_replica_loss = self._distribution_strategy.experimental_run_v2(
                self._train_step if step else self._val_step, args=(batch,))
            total_loss += self._distribution_strategy.reduce(
                tf.distribute.ReduceOp.SUM, per_replica_loss,
                axis=None)
            num_batches += 1.0
            tf.print(num_batches, ':', total_loss / num_batches, sep='')
        total_loss = total_loss / num_batches
        return total_loss

    def _configure_callbacks(self, callbacks):
        for callback in callbacks:
            callback.set_model(self)

    def fit(
            self,
            epochs,
            callbacks,
            train_dataset,
            val_dataset,
            writer=None,
            use_ema=False,
            use_adv=False,
            adv_config=nsl.configs.make_adv_reg_config(
                multiplier=0.2, adv_step_size=0.2, adv_grad_norm='infinity'),
    ):
        self.writer = writer
        self.use_ema = use_ema
        self.use_adv = use_adv
        self.adv_config = adv_config
        self._configure_callbacks(callbacks)
        logs = {}
        self.stop_training=False
        for callback in callbacks:
            callback.on_train_begin(logs)
        for epoch in range(epochs):
            if self.stop_training:
                break
            for callback in callbacks:
                callback.on_epoch_begin(epoch, logs)
            train_loss = self._distributed_epoch(
                train_dataset, True)
            val_loss = self._distributed_epoch(
                val_dataset, False)
            logs['loss'] = train_loss
            logs['val_loss'] = val_loss
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)

        for callback in callbacks:
            callback.on_train_end(logs)