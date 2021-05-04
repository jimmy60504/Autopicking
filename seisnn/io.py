"""
Input / Output
"""

import collections
import multiprocessing as mp
import itertools
import os
import warnings

from lxml import etree
from obspy import Stream
from obspy.core import inventory
from obspy.clients.filesystem import sds
import obspy.io.nordic.core
import obspy
import tensorflow as tf

import seisnn
import seisnn.example_proto
import seisnn.utils


def read_dataset(file_list):
    """
    Returns TFRecord Dataset from TFRecord directory.

    :param file_list: List of .tfrecord.
    :rtype: tf.data.Dataset
    :return: A Dataset.
    """

    dataset = tf.data.TFRecordDataset(file_list)
    dataset = dataset.map(seisnn.example_proto.sequence_example_parser,
                          num_parallel_calls=mp.cpu_count())
    return dataset


def write_tfrecord(example_list, save_file):
    """
    Writes TFRecord from example protocol.

    :param list example_list: List of example protocol.
    :param save_file: Output file path.
    """
    with tf.io.TFRecordWriter(save_file) as writer:
        for example in example_list:
            writer.write(example)


def read_event_list(sfile_dir):
    """
    Returns event list from sfile directory.

    :param str sfile_dir: Directory contains SEISAN sfile.
    :rtype: list
    :return: list of event.
    """
    config = seisnn.utils.Config()
    sfile_dir = os.path.join(config.catalog, sfile_dir)

    sfile_list = seisnn.utils.get_dir_list(sfile_dir, ".S*")
    print(f'Reading events from {sfile_dir}')

    event_list = seisnn.utils.parallel(sfile_list, func=get_event)
    flatten = itertools.chain.from_iterable

    events = list(flatten(flatten(event_list)))
    print(f'Read {len(events)} events\n')
    return events


def get_event(file, debug=False):
    """
    Returns obspy.event list from sfile.

    :param str file: Sfile file path.
    :param bool debug: If False, warning from reader will be ignore,
        default to False.
    :rtype: list
    :return: List of events.
    """
    with warnings.catch_warnings():
        if not debug:
            warnings.simplefilter("ignore")
        try:
            catalog = obspy.io.nordic.core.read_nordic(file)
            return catalog.events

        except Exception as err:
            if debug:
                print(err)


def read_sds(metadata):
    """
    Read SDS database.

    :param metadata: Metadata.
    :rtype: dict
    :return: Dict contains all traces within the time window.
    """
    config = seisnn.utils.Config()
    station = metadata.station
    starttime = metadata.starttime
    endtime = metadata.endtime + 0.1

    client = sds.Client(sds_root=config.sds_root)
    stream = client.get_waveforms(network="*",
                                  station=station,
                                  location="*",
                                  channel="*",
                                  starttime=starttime,
                                  endtime=endtime)
    stream.sort(keys=['channel'], reverse=True)

    stream_dict = collections.defaultdict(Stream)
    for trace in stream:
        geophone_type = trace.stats.channel[0:2]
        stream_dict[geophone_type].append(trace)

    return stream_dict


def read_hyp(hyp):
    """
    Returns geometry from STATION0.HYP file.

    :param str hyp: STATION0.HYP name without directory.
    :rtype: dict
    :return: Geometry dict.
    """
    config = seisnn.utils.Config()
    hyp_file = os.path.join(config.geom, hyp)
    geom = {}
    with open(hyp_file, 'r') as file:
        blank_line = 0
        while True:
            line = file.readline().rstrip()

            if not len(line):
                blank_line += 1
                continue

            if blank_line > 1:
                break

            elif blank_line == 1:
                lat = line[6:14]
                lon = line[14:23]
                elev = float(line[23:])
                sta = line[1:6].strip()

                NS = 1
                if lat[-1] == 'S':
                    NS = -1

                EW = 1
                if lon[-1] == 'W':
                    EW = -1

                lat_degree = int(lat[0:2])
                lat_minute = float(lat[2:-1]) / 60
                if '.' not in lat:  # high accuracy lat-lon
                    lat_minute /= 1000
                lat = (lat_degree + lat_minute) * NS
                lat = inventory.util.Latitude(lat)

                lon_degree = int(lon[0:3])
                lon_minute = float(lon[3:-1]) / 60
                if '.' not in lon:  # high accuracy lat-lon
                    lon_minute /= 1000
                lon = (lon_degree + lon_minute) * EW
                lon = inventory.util.Longitude(lon)

                location = {'latitude': lat,
                            'longitude': lon,
                            'elevation': elev}
                geom[sta] = {'location': location}
    print(f'read {len(geom)} stations from {hyp}')
    return geom


def read_nsta(nsta):
    config = seisnn.utils.Config()
    hyp_file = os.path.join(config.geom, nsta)
    geom = {}
    f = open(hyp_file, 'r')
    for line in f:
        sta = line[0:5].strip()
        lon = float(line[5:13])
        lat = float(line[14:21])
        elev = float(line[23:30])
        net = line[34:37].strip()

        lat = inventory.util.Latitude(lat)
        lon = inventory.util.Longitude(lon)

        location = {'latitude': lat,
                    'longitude': lon,
                    'elevation': elev}

        geom[sta] = {'network': net,
                     'location': location}
    print(f'read {len(geom)} stations from {nsta}')
    return geom


def write_hyp_station(geom, save_file):
    """
    Write STATION0.HYP file from geometry.

    :param dict geom: Geometry dict.
    :param str save_file: Name of .HYP file.
    """
    config = seisnn.utils.Config()
    hyp = []
    for sta, loc in geom.items():
        lat = int(loc['latitude'])
        lat_min = (loc['latitude'] - lat) * 60

        NS = 'N'
        if lat < 0:
            NS = 'S'

        lon = int(loc['longitude'])
        lon_min = (loc['longitude'] - lon) * 60

        EW = 'E'
        if lat < 0:
            EW = 'W'

        elev = int(loc['elevation'])

        hyp.append(
            f' {sta: >5}{lat: >2d}{lat_min:>5.2f}{NS}{lon: >3d}{lon_min:>5.2f}{EW}{elev: >4d}\n')
    hyp.sort()

    output = os.path.join(config.geom, save_file)
    with open(output, 'w') as f:
        f.writelines(hyp)


def read_kml_placemark(kml):
    """
    Returns geometry from Google Earth KML file.

    :param str kml: KML file name without directory.
    :rtype: dict
    :return: Geometry dict.
    """
    config = seisnn.utils.Config()
    kml_file = os.path.join(config.geom, kml)

    parser = etree.XMLParser()
    root = etree.parse(kml_file, parser).getroot()
    geom = {}
    for Placemark in root.findall('.//Placemark', root.nsmap):
        sta = Placemark.find('.//name', root.nsmap).text
        coord = Placemark.find('.//coordinates', root.nsmap).text
        coord = coord.split(",")
        location = {'latitude': float(coord[1]),
                    'longitude': float(coord[0]),
                    'elevation': float(coord[2])}
        geom[sta] = location

    print(f'read {len(geom)} stations from {kml}')
    return geom


if __name__ == "__main__":
    pass


def read_header(header):
    header_info = {}
    header_info['year'] = int(header[1:5])
    header_info['month'] = int(header[5:7])
    header_info['day'] = int(header[7:9])
    header_info['hour'] = int(header[9:11])
    header_info['minute'] = int(header[11:13])
    header_info['second'] = float(header[13:19])
    header_info['lat'] = float(header[19:21])
    header_info['lat_minute'] = float(header[21:26])
    header_info['lon'] = int(header[26:29])
    header_info['lon_minute'] = float(header[29:34])
    header_info['depth'] = float(header[34:40])
    header_info['magnitude'] = float(header[40:44])
    header_info['nsta'] = header[44:46].replace(" ", "")
    header_info['Pfilename'] = header[46:58].replace(" ", "")
    header_info['newNoPick'] = header[60:63].replace(" ", "")
    return header_info


def read_lines(lines):
    trace = []
    for line in lines:
        try:
            line_info = {}
            line_info['code'] = str(line[1:7]).replace(" ", "")
            line_info['epdis'] = float(line[7:13])
            line_info['az'] = int(line[13:17])
            line_info['phase'] = str(line[21:22]).replace(" ", "")
            line_info['ptime'] = float(line[23:30])
            line_info['pwt'] = int(line[30:32])
            line_info['stime'] = float(line[33:40])
            line_info['swt'] = int(line[40:42])
            line_info['lat'] = float(line[42:49])
            line_info['lon'] = float(line[49:57])
            line_info['gain'] = float(line[57:62])
            line_info['convm'] = str(line[62:63]).replace(" ", "")
            line_info['accf'] = str(line[63:75]).replace(" ", "")
            line_info['durt'] = float(line[75:79])
            line_info['cherr'] = int(line[80:83])
            line_info['timel'] = str(line[83:84]).replace(" ", "")
            line_info['rtcard'] = str(line[84:101]).replace(" ", "")
            line_info['ctime'] = str(line[101:109]).replace(" ", "")
        except ValueError:
            print(line)
            continue
        trace.append(line_info)

    return trace


def read_afile(afile_path):
    count = 0
    f = open(afile_path, 'r')
    header = f.readline()
    lines = f.readlines()
    header_info = read_header(header)
    trace_info = read_lines(lines)
    # event_info['header_info'] = header_info
    # event_info['trace_info'] = trace_info
    # count = len(trace_info)
    ev = obspy.core.event.Event()
    ev.event_descriptions.append(obspy.core.event.EventDescription())
    ev.origins.append(obspy.core.event.Origin(
        time=obspy.UTCDateTime(header_info['year'], header_info['month'],
                               header_info['day'], header_info['hour'],
                               header_info['minute'], header_info['second']),
        latitude=header_info['lat'] + header_info['lat_minute'] / 60,
        longitude=header_info['lon'] + header_info['lon_minute'] / 60,
        depth=header_info['depth']))
    for trace in trace_info:
        _waveform_id_1 = obspy.core.event.WaveformStreamID(
            station_code=trace['code'],
            channel_code='',
            network_code=''
        )
        for phase in ['P', 'S']:
            ev.picks.append(
                obspy.core.event.Pick(waveform_id=_waveform_id_1,
                                      phase_hint=phase,
                                      time=obspy.UTCDateTime(
                                          header_info['year'],
                                          header_info['month'],
                                          header_info['day'],
                                          header_info['hour'],
                                          header_info['minute'],
                                          header_info['second']) + trace[
                                               f'{phase.lower()}time']

                                      )
            )
            count += 1

    return ev, count


def read_afile_directory(path_list):
    event_list = []
    trace_count = 0
    abs_path = seisnn.utils.get_dir_list(path_list)
    for path in abs_path:
        try:
            event, c = read_afile(path)
            event_list.append(event)
            trace_count += c
        except ValueError:
            continue

    print('total_pick = ', trace_count)
    return event_list
