"""
Training settings.
"""
import abc
import os
import shutil

import tensorflow as tf

from seisnn.model.generator import nest_net
from seisnn.data import example_proto, io, logger, sql
from seisnn.data.core import Instance
from seisnn import utils


class BaseTrainer(abc.ABC):
    @staticmethod
    def get_dataset_length(database):
        db = sql.Client(database)
        with db.session_scope() as session:
            Waveform = db.get_table_class('waveform')
            count = session.query(Waveform).count()

        return count

    @staticmethod
    def create_model_folder(model_instance, remove=False):
        config = utils.get_config()
        save_model_path = os.path.join(config['MODELS_ROOT'], model_instance)

        if remove:
            shutil.rmtree(save_model_path, ignore_errors=True)
        utils.make_dirs(save_model_path)

        save_history_path = os.path.join(save_model_path, "history")
        utils.make_dirs(save_history_path)

        return save_model_path, save_history_path

    def train_step(self, train, val):
        pass

    def train_loop(self,
                   dataset,
                   model_name,
                   epochs,
                   batch_size,
                   plot=False):
        pass


class GeneratorTrainer(BaseTrainer):
    """
    Trainer class.
    """

    def __init__(self,
                 database=None,
                 model=nest_net(),
                 optimizer=tf.keras.optimizers.Adam(1e-4),
                 loss=tf.keras.losses.BinaryCrossentropy()):
        """
        Initialize the trainer.

        :param database: SQL database.
        :param model: keras model.
        :param optimizer: keras optimizer.
        :param loss: keras loss.
        """
        self.database = database
        self.model = model
        self.optimizer = optimizer
        self.loss = loss

    @tf.function
    def train_step(self, train, val):
        """
        Training step.

        :param train: Training data.
        :param val: Validation data.
        :rtype: float
        :return: predict loss, validation loss
        """
        with tf.GradientTape(persistent=True) as tape:
            train_pred = self.model(train['trace'], training=True)
            train_loss = self.loss(train['label'], train_pred)

            val_pred = self.model(val['trace'], training=False)
            val_loss = self.loss(val['label'], val_pred)

            gradients = tape.gradient(train_loss,
                                      self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))

            return train_loss, val_loss

    def train_loop(self,
                   dataset, model_name,
                   epochs=1, batch_size=1,
                   log_step=100, plot=False):
        """
        Main training loop.

        :param str dataset: Dataset name.
        :param str model_name: Model directory name.
        :param int epochs: Epoch number.
        :param int batch_size: Batch size.
        :param int log_step: Logging step interval.
        :param bool plot: Plot training validation,
            False save fig, True show fig.
        :return:
        """
        model_path, history_path = self.create_model_folder(model_name)
        dataset = io.read_dataset(dataset).shuffle(100000)
        val = next(iter(dataset.batch(1)))

        ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, model_path,
                                                  max_to_keep=100)

        metrics_names = ['loss', 'val']
        progbar = tf.keras.utils.Progbar(
            self.get_dataset_length(self.database),
            stateful_metrics=metrics_names)

        for epoch in range(epochs):
            print(f'epoch {epoch + 1} / {epochs}')
            n = 0
            loss_buffer = []
            for train in dataset.prefetch(100).batch(batch_size):
                train_loss, val_loss = self.train_step(train, val)
                loss_buffer.append([train_loss, val_loss])

                values = [('loss', train_loss.numpy()),
                          ('val', val_loss.numpy())]
                progbar.add(batch_size, values=values)

                if n % log_step == 0:
                    logger.save_loss(loss_buffer, model_name, model_path)
                    loss_buffer.clear()

                    title = f'epoch{epoch + 1:0>2}_step{n:0>5}___'
                    val['predict'] = self.model.predict(val['trace'])
                    val['id'] = tf.convert_to_tensor(
                        title.encode('utf-8'), dtype=tf.string)[tf.newaxis]

                    example = next(example_proto.batch_iterator(val))
                    instance = Instance(example)

                    if plot:
                        instance.plot()
                    else:
                        instance.plot(save_dir=history_path)
                n += 1

        ckpt_save_path = ckpt_manager.save()
        print(f'Saving pre-train checkpoint to {ckpt_save_path}')
