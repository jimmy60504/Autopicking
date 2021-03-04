"""
Core
"""
import itertools
import os

import numpy as np
import scipy.signal
import obspy

import seisnn.example_proto
import seisnn.io
import seisnn.plot
import seisnn.sql


class Metadata:
    """
    Main class for metadata.
    """
    id = None
    station = None

    starttime = None
    endtime = None
    npts = None
    delta = None

    data = None

    def from_trace(self, trace):
        self.id = trace.id
        self.station = trace.stats.station

        self.starttime = trace.stats.starttime
        self.endtime = trace.stats.endtime
        self.npts = trace.stats.npts
        self.delta = trace.stats.delta
        return self

    def from_feature(self, feature):
        self.id = feature['id']
        self.station = feature['station']

        self.starttime = obspy.UTCDateTime(feature['starttime'])
        self.endtime = obspy.UTCDateTime(feature['endtime'])
        self.npts = feature['npts']
        self.delta = feature['delta']
        return self


class Trace(Metadata):
    """
    Main class for trace data.
    """
    channel = None
    data = None
    metadata = None

    def from_stream(self, stream):
        """
        Gets waveform from Obspy stream.

        :param stream: Obspy stream object.
        :return: Waveform object.
        """
        channel = []
        trace = np.zeros([3008, 3])
        for i, comp in enumerate(['Z', 'N', 'E']):
            try:
                st = stream.select(component=comp)
                trace[:, i] = st.traces[0].data
                channel.append(st.traces[0].stats.channel)

            except IndexError:
                pass

            except Exception as error:
                print(f'{type(error).__name__}: {error}')

        self.data = trace
        self.channel = channel
        self.metadata = Metadata().from_trace(stream.traces[0])

        return self

    def from_feature(self, feature):
        self.metadata = Metadata().from_feature(feature)
        self.data = feature['trace']
        self.channel = feature['channel']
        return self


class Label(Metadata):
    """
    Main class for label data.
    """
    phase = None
    data = None
    metadata = None

    picks = None

    def generate_label(self, database, tag, shape, half_width=20):
        """
        Add generated label to stream.

        :param str database: SQL database.
        :param str tag: Pick tag in database.
        :param str shape: Label shape, see scipy.signal.windows.get_window().
        :param int half_width: Label half width in data point.
        :rtype: np.array
        :return: Label.
        """
        db = seisnn.sql.Client(database)
        label = np.zeros([self.metadata.npts, len(self.phase)])

        ph_index = {}
        for i, phase in enumerate(self.phase):
            ph_index[phase] = i
            picks = db.get_picks(from_time=self.metadata.starttime.datetime,
                                 to_time=self.metadata.endtime.datetime,
                                 station=self.metadata.station,
                                 phase=phase, tag=tag).all()

            for pick in picks:
                pick_time = obspy.UTCDateTime(
                    pick.time) - self.metadata.starttime
                pick_time_index = int(pick_time / self.metadata.delta)
                label[pick_time_index, i] = 1

        if 'EQ' in self.phase:
            # Make EQ window start by P and end by S.
            label[:, ph_index['EQ']] = label[:, ph_index['P']] \
                                       - label[:, ph_index['S']]
            label[:, ph_index['EQ']] = np.cumsum(label[:, ph_index['EQ']])
            if np.any(label[:, ph_index['EQ']] < 0):
                label[:, ph_index['EQ']] += 1

        for i, phase in enumerate(self.phase):
            if not phase == 'EQ':
                wavelet = scipy.signal.windows.get_window(shape,
                                                          2 * half_width)
                label[:, i] = scipy.signal.convolve(label[:, i], wavelet[1:],
                                                    mode='same')

        if 'N' in self.phase:
            # Make Noise window by 1 - P - S
            label[:, ph_index['N']] = 1
            label[:, ph_index['N']] -= label[:, ph_index['P']]
            label[:, ph_index['N']] -= label[:, ph_index['S']]

        return self

    def get_picks(self, height=0.4, distance=100):
        """
        Extract pick from label and write into the database.

        :param float height: Height threshold, from 0 to 1, default is 0.5.
        :param int distance: Distance threshold in data point.
        """
        picks = []
        for i, phase in enumerate(self.phase[0:2]):
            peaks, _ = scipy.signal.find_peaks(
                self.data[-1, :, i],
                height=height,
                distance=distance)

            for peak in peaks:
                if peak:
                    pick_time = obspy.UTCDateTime(self.starttime) \
                                + peak * self.delta

                    picks.append(Pick(time=pick_time,
                                      station=self.station,
                                      phase=self.phase[i])
                                 )

        self.picks = picks

    def write_picks_to_database(self, tag, database):
        """
        Write picks into the database.

        :param str tag: Output pick tag name.
        :param database: SQL database name.
        """
        db = seisnn.sql.Client(database)
        for pick in self.picks:
            db.add_pick(time=pick.time.datetime,
                        station=pick.station,
                        phase=pick.phase,
                        tag=tag)


class Pick:
    """
    Main class for phase pick.
    """

    def __init__(self,
                 time=None,
                 station=None,
                 phase=None,
                 tag=None):

        self.time = time
        self.station = station
        self.phase = phase
        self.tag = tag


class Instance:
    """
    Main class for data transfer.
    """
    metadata = None

    trace = None
    label = None
    predict = None

    def __init__(self, input_data=None):
        if input_data is None:
            pass
        try:
            if isinstance(input_data, seisnn.sql.Waveform):
                dataset = seisnn.io.read_dataset(input_data.dataset)
                for item in dataset.skip(input_data.data_index).take(1):
                    input_data = item

            self.from_example(input_data)
        except TypeError:
            pass

        except Exception as error:
            print(f'{type(error).__name__}: {error}')

    def __repr__(self):
        return f"Instance(" \
               f"ID={self.metadata.id}, " \
               f"Start Time={self.metadata.starttime}, " \
               f"Phase={self.phase})"

    def from_stream(self, stream):
        """

        :param stream:
        :return:
        """
        self.trace = Trace().from_stream(stream)
        self.metadata = self.trace.metadata
        return self

    def from_feature(self, feature):
        """
        Initialized from feature dict.

        :type feature: dict
        :param feature: Feature dict.
        """
        self.trace = Trace().from_feature(feature)
        self.metadata = self.trace.metadata

        self.phase = feature['phase']
        self.label = feature['label']
        self.predict = feature['predict']

    def to_feature(self):
        """
        Returns feature dict.

        :rtype: dict
        :return: Feature dict.
        """
        feature = {
            'id': self.metadata.id,
            'station': self.metadata.station,
            'starttime': self.metadata.starttime.isoformat(),
            'endtime': self.metadata.endtime.isoformat(),

            'npts': self.metadata.npts,
            'delta': self.metadata.delta,

            'trace': self.trace.data,
            'channel': self.trace.channel,

            'phase': self.phase,
            'label': self.label,
            'predict': self.predict,
        }
        return feature

    def from_example(self, example):
        """
        Initialized from example protocol.

        :param example: Example protocol.
        """
        feature = seisnn.example_proto.eval_eager_tensor(example)
        self.from_feature(feature)

    def to_example(self):
        """
        Returns example protocol.

        :return: Example protocol.
        """
        feature = self.to_feature()
        example = seisnn.example_proto.feature_to_example(feature)
        return example

    def to_tfrecord(self, file_path):
        """
        Write TFRecord to file path.

        :param str file_path: Output path.
        """
        feature = self.to_feature()
        example = seisnn.example_proto.feature_to_example(feature)
        seisnn.io.write_tfrecord([example], file_path)

    def plot(self, **kwargs):
        """
        Plot dataset.

        :param kwargs: Keywords pass into plot.
        """
        seisnn.plot.plot_dataset(self, **kwargs)


class ExampleGen:
    """
    Main class for Example Generator.

    Consumes data from external source and emit TFRecord.
    """
    phase = ['P', 'S', 'N']
    trace_length = 30
    points = 3008

    def generate_training_data(self,
                               pick_list,
                               dataset,
                               tag,
                               database,
                               chunk_size=64):
        """
        Generate TFRecords from database.

        :param pick_list: List of picks from Pick SQL query.
        :param str dataset: Output directory name.
        :param str database: SQL database name.
        :param int chunk_size: Number of data stores in TFRecord.
        """
        config = seisnn.utils.get_config()
        dataset_dir = os.path.join(config['DATASET_ROOT'], dataset)
        seisnn.utils.make_dirs(dataset_dir)

        total_batch = int(len(pick_list) / chunk_size)
        batch_picks = seisnn.utils.batch(pick_list, size=chunk_size)
        for index, picks in enumerate(batch_picks):
            example_list = seisnn.utils.parallel(picks,
                                                 func=self.get_example_list,
                                                 tag=tag,
                                                 database=database)
            flatten = itertools.chain.from_iterable
            flat_list = list(flatten(flatten(example_list)))

            file_name = f'{index:0>5}.tfrecord'
            save_file = os.path.join(dataset_dir, file_name)
            seisnn.io.write_tfrecord(flat_list, save_file)
            print(f'output {file_name} / {total_batch}')

    def get_example_list(self, pick, tag, database):
        """
        Returns example list form list of picks and SQL database.

        :param pick: List of picks.
        :param str database: SQL database root.
        :return:
        """

        metadata = self.get_time_window(anchor_time=pick.time,
                                        station=pick.station,
                                        shift='random')

        streams = seisnn.io.read_sds(metadata)
        example_list = []
        for _, stream in streams.items():
            stream = self.signal_preprocessing(stream)

            instance = seisnn.core.Instance(stream)
            instance.phase = self.phase
            instance.label.generate_label(database, tag,
                                          shape='triang')
            instance.predict = np.zeros(instance.label.data.shape)

            feature = instance.to_feature()
            example = seisnn.example_proto.feature_to_example(feature)
            example_list.append(example)
        return example_list

    def get_time_window(self, anchor_time, station, shift=0):
        """
        Returns time window from anchor time.

        :param anchor_time: Anchor of the time window.
        :param str station: Station name.
        :param float or str shift: (Optional.) Shift in sec,
            if 'random' will shift randomly within the trace length.
        :rtype: dict
        :return: Time window.
        """
        if shift == 'random':
            rng = np.random.default_rng()
            shift = rng.random() * self.trace_length

        metadata = Metadata()
        metadata.starttime = obspy.UTCDateTime(anchor_time) - shift
        metadata.endtime = metadata.starttime + self.trace_length
        metadata.station = station

        return metadata

    def signal_preprocessing(self, stream):
        """
        Return a signal processed stream.

        :param obspy.Stream stream: Stream object.
        :rtype: obspy.Stream
        :return: Processed stream.
        """
        stream.detrend('demean')
        stream.detrend('linear')
        stream.normalize()
        stream.resample(100)
        stream = self.trim_trace(stream)
        return stream

    def trim_trace(self, stream, points=3008):
        """
        Return trimmed stream in a given length.

        :param obspy.Stream stream: Stream object.
        :param int points: Trace data length.
        :rtype: obspy.Stream
        :return: Trimmed stream.
        """

        trace = stream[0]
        start_time = trace.stats.starttime
        dt = (trace.stats.endtime - trace.stats.starttime) / (
                trace.data.size - 1)
        end_time = start_time + dt * (points - 1)
        stream.trim(start_time,
                    end_time,
                    nearest_sample=True,
                    pad=True,
                    fill_value=0)
        return stream


if __name__ == "__main__":
    pass
