import argparse
import os
import sys
from datetime import datetime
from math import ceil, floor
from pathlib import Path
from typing import List, Dict

import librosa

from typing import List
from abc import ABCMeta
# from BNResultTools.RelativeTimeSegment import LabeledRelativeTimeSegment

def to_mm_ss(seconds: float) -> str:
    """

    :param seconds:
    :return: the string representation in format *mm:ss with leading zeros
    """
    m = int(seconds / 60)
    s = seconds % 60

    ret = str(m) + ":"
    if m < 10:
        ret = "0" + ret
    if s < 10:
        ret += "0"
    ret += str(s).rstrip('.')
    return ret

class RelativeTimeSegment(metaclass=ABCMeta):
    """
        Represents a continuous not absolute time segment, in seconds
    """

    def __init__(self, t0: float, t1: float):
        if t0 <= t1:
            self.begin_time = t0
            self.end_time = t1
        else:
            self.begin_time = t1
            self.end_time = t0

    def extended(self, additional_seconds: float) -> 'RelativeTimeSegment':
        """
        Creates a new RelativeTimeSegment extendend symetrically by the 'additional_seconds' time.
        The begin_time of the new segment will be at least 0 (never negative)

        :param additional_seconds:
        :return: the new, extended RelativeTimeSegment
        """
        nb:float = self.begin_time - additional_seconds;
        if nb < 0:
            nb = 0.0
        ne: float = self.end_time + additional_seconds
        return RelativeTimeSegment(nb, ne)


    def overlaping_time(self, other: 'RelativeTimeSegment') -> float:
        """

        :param other: another segment
        :return: the length of the overlapping time, in seconds
        """
        if self.end_time <= other.begin_time:
            return 0.0
        if self.begin_time >= other.end_time:
            return 0.0
        ov_b = max(self.begin_time, other.begin_time)
        ov_e = min(self.end_time, other.end_time)
        return ov_e - ov_b

    def duration(self) -> float:
        return self.end_time - self.begin_time

    def middle(self) -> float:
        return self.end_time / 2.0 + self.begin_time / 2.0;

    def to_mmss_string(self, separator: str = "-") -> str:
        return to_mm_ss(self.begin_time) + separator + to_mm_ss(self.end_time)
    def to_int_mmss_string(self, separator: str = "-") -> str:
        return to_mm_ss(math.floor(self.begin_time)) + separator + to_mm_ss(math.ceil(self.end_time))
    def to_string(self, separator: str = "-") -> str:
        return str(self.begin_time) + separator + str(self.end_time)

    # operator  << : move the timesegment to 'earlier'
    def __lshift__(self, period: float) -> 'RelativeTimeSegment':
        return RelativeTimeSegment(max(0.0, self.begin_time-period), max(0.0, self.end_time-period))

    #operator >> : move the timesegment to 'later'
    def __rshift__(self, period:float) -> 'RelativeTimeSegment':
        return self << (-period)


class LabeledRelativeTimeSegment(RelativeTimeSegment):

    def __init__(self, t0: float, t1: float, label: str ="") -> 'LabeledRelativeTimeSegment':
        self.label = label
        RelativeTimeSegment.__init__(self, t0, t1)

    @staticmethod
    def from_rts(rts: RelativeTimeSegment, label: str ) -> 'LabeledRelativeTimeSegment':
        return LabeledRelativeTimeSegment( rts.begin_time, rts.end_time, label)


class AudacityLabel(LabeledRelativeTimeSegment):
    """
        Represents data of a single line of the standard Audacity label file.
        The fields:

        begin time (s) <tab> end time (s) <tab> label

    """

    def __init__(self, t0: float, t1: float, label: str):
        LabeledRelativeTimeSegment.__init__(self, t0, t1, label)

    def __init__(self, raw_record: List[str]) -> None:
        begin_time = float(raw_record[0])
        end_time = float(raw_record[1])
        label = raw_record[2].strip()
        LabeledRelativeTimeSegment.__init__(self, begin_time, end_time, label)
        # SimpleLabel.__init__(label)

    # tab separated, no header line
    # 0 begin
    # 1 end
    # 2 label

    @staticmethod
    def parse_file(filename: str) -> List['AudacityLabel']:
        """
        Reads a single label file into a list of records
        :param filename: full path of the file to read
        :return:
        """
        records = []
        file = open(filename, "r")
        curr = 0
        for line in file.readlines():
            curr += 1
            data = line.split("\t")
            for i in range(0, len(data)):
                data[i] = data[i].strip()
            current = AudacityLabel(data)
            records.append(current)
        return records

def list_of_files(dir: Path, extension: str) -> List[Path]:
    return list_of_files_ex(dir, [extension])

def list_of_files_ex(dir: Path, extensions: List[str]) -> List[Path]:
    all_files: List[Path] = []
    for dirPath, dirNames, fileNames in os.walk(dir):
        for f in fileNames:
            for ex in extensions:
                if f.lower().endswith(ex.lower()):
                    all_files.append(Path(dirPath) / f)
    return all_files

def dictionary_by_bare_name(files: List[Path], suffix_to_remove: str) -> Dict[str, Path]:
    #    print("checking the files"+str(files))
    d: Dict[str, Path] = {}
    for f in files:
        if f.name.lower().endswith(suffix_to_remove.lower()):
            bn = f.name[0:-len(suffix_to_remove)]
            #print("[" + bn + "] -> " + str(f))
            if not d.get(bn, False):
                d[bn] = f
            else:
                sys.stderr.write("[" + bn + "] is not a unique name: \n" + str(f) + "\nor\n" +  str(d[bn]) + " - kept as first")
        else:
            sys.stderr.write(str(f) + " doesn't have the expecetd suffix [" + suffix_to_remove + "]")
    return d


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def parse_commandline():
    parser = argparse.ArgumentParser()
    parser.description = f'Extracts audio chunks from long wav file based on Audacity label file. The result consists ' \
                         f'of multiple audio file, each 3s long \'chunk\', and a one-hot_labels.csv describing ' \
                         f'presence or absence of the annotated call in all the chunks'
    parser.add_argument("-i", "--input_dir",
                        help="Path to the file or folder of the manual annotation in the *.txt (Audacity) format",
                        default=".")
    parser.add_argument("-o", "--output-directory",
                        help="Path to the output directory. If doesn't exist, it will be created.",
                        default="chunks_from_labels")
    # parser.add_argument("-a", "--annotation",
    #                     help="Uses only these lines from the input file(s) that contain this string", default="")
    parser.add_argument("-r", "--resample", help="Resample the chunk to the given value in Hz.", type=int,
                        default=24000)
    return parser.parse_args()


def createChunks(cd: RelativeTimeSegment, audio_file: Path, name_stem: str, output_dir: Path,
                 annotation: str, sample_rate: int) -> bool:
    audio_type: str = 'flac' #"wav"  # "ogg" # "mp3"
    a = annotation.replace("?", "_MAYBE")

    new_file_name = name_stem + "_" + cd.to_int_mmss_string().replace(":", ";") + "_(" + a + ")." + audio_type
    new_file_path: Path = output_dir / new_file_name
    ret: bool = createChunkFile(new_file_path, cd, audio_file, sample_rate)

    return ret, new_file_path

    # cd_p3: RelativeTimeSegment = cd.extended(3)
    # nfp_p3: Path = output_dir / Path(
    #     name_stem + "_" + cd.to_int_mmss_string().replace(":", ";") + "_(" + a + ")_3s_extended." + audio_type)
    # ret_p3: bool = createChunkFile(nfp_p3, cd_p3, audio_file)
    #
    # return ret and ret_p3, new_file_path


def createChunkFile(file_path: Path, rts: RelativeTimeSegment, audio_file: Path, sample_rate: int) -> bool:
    if file_path.exists():
        print("The file  [" + str(file_path) + "] already exists, skipping creating", file=sys.stderr)
        return False
    sys_call: str = "sox \"" + str(audio_file) + "\" \"" + str(file_path) + "\"" + " trim " + str(
        rts.begin_time) + " " + str(rts.duration()) + " " + f'rate {sample_rate}'
    print(sys_call);
    exit_code = os.system(sys_call)
    if exit_code != 0:
        print("Error! The command  [" + sys_call + "] returned [" + str(exit_code) + "]", file=sys.stderr)
        return False
    return True


def create_output_dir_structure(results_folder: Path) -> (Path, Path):
    dt: datetime = datetime.utcnow()
    prefix: str = dt.strftime("%Y%m%d_%H%M%SUTC")
    output_type: str = "chunks"
    out_dir = results_folder / Path(f"{prefix}_{output_type}")
    os.mkdir(out_dir)
    if not out_dir.is_dir():
        raise NotADirectoryError("Cannot create output directory " + str(out_dir))
    report_file: Path = out_dir / Path(f'one-hot_labels.csv')
    return out_dir, report_file


def generate_chunk_coverage(ts: LabeledRelativeTimeSegment, chunk_length: float, overlap: float) -> List[LabeledRelativeTimeSegment]:
    if ts.duration()<chunk_length:
        return [ts]
    if ts.duration() - chunk_length < 1.0:
        return [LabeledRelativeTimeSegment(ts.middle() - chunk_length/2.0, ts.middle()+chunk_length/2.0, ts.label)]
    step: float = chunk_length - overlap
    n_steps: int = ceil((ts.duration() - chunk_length) / step)
    real_step: float = (ts.duration() - chunk_length) / n_steps
    results_list: List[LabeledRelativeTimeSegment] = []
    first_ts: LabeledRelativeTimeSegment = LabeledRelativeTimeSegment(ts.begin_time, ts.begin_time+chunk_length, ts.label)
    for i in range(0, n_steps+1, 1):
        results_list.append( LabeledRelativeTimeSegment.from_rts(first_ts >> (i * real_step),ts.label))
    return results_list

def get_annotation_labels(fp_list:List[Path]) -> List[str]:
    labels: Dict[str,int] = {}
    for fp in fp_list:
        try:
            chunk_definitions: List[AudacityLabel] = AudacityLabel.parse_file(fp)
            for cd in chunk_definitions:
                labels[cd.label.strip().lower()] = 1
        except:
            pass
    return labels.keys()


def do_the_butchery(args):
    chunk_length: int = 2.0
    input_extension = ".txt" #".Table.1.selections.txt"  # ".wav.csv"
    chunk_def_files = list_of_files(args.input_dir, input_extension)
    af_extension = ".flac" #, ".wav"
    audio_files = dictionary_by_bare_name(list_of_files(args.input_dir, af_extension), af_extension)
    # print(str(audio_files))
    odir: Path = Path(args.output_directory)
    if not odir.exists():
        os.mkdir(odir)
    exe_output_dir, report_file = create_output_dir_structure(odir)
    with open(report_file, "w") as rf:
        df_processed = 0
        print("Extracting audio fragments report file. \n")
        print(f"Input folder: {args.input_dir}")
        print(f"\nFound {len(chunk_def_files)} inpupt files.\n")
        classes: List[str] = ["_nothing"]
        classes.extend(get_annotation_labels(chunk_def_files))
        print(f'filename, ' + ','.join(classes), file=rf)
        for cdf in chunk_def_files:
            try:
                chunk_definitions: List[AudacityLabel] = AudacityLabel.parse_file(cdf)
                c_count = 0
                not_annot = 0
                name_stem: str = str(cdf.name)[0:-len(input_extension)-4]  # remove the extension and the _AGC or _SDI
                audio_file = audio_files.get(name_stem, None)
                if audio_file is None:
                    raise FileNotFoundError(name_stem)
                duration = librosa.get_duration(filename=audio_file)
                if duration < chunk_length:
                    raise RuntimeError(f'audio file {name_stem} is too short ({duration}s) ')
                not_annotated: List[LabeledRelativeTimeSegment] = []
                for i in range(0, floor(duration/chunk_length )):
                    not_annotated.append( LabeledRelativeTimeSegment( i * chunk_length,(i+1)*chunk_length, classes[0]) )

                annotated: List[LabeledRelativeTimeSegment] = []
                for cd in chunk_definitions:
                    for nt_label in not_annotated:
                        if nt_label.overlaping_time(cd) > 0 :
                            not_annotated.remove(nt_label)
                    annotated.extend(generate_chunk_coverage(ts = LabeledRelativeTimeSegment.from_rts(cd.extended(1), cd.label), chunk_length=chunk_length, overlap=1.5) )

                for chunk in annotated + not_annotated:
                    success, name = createChunks(chunk, audio_file, name_stem, exe_output_dir, chunk.label, args.resample)
                    one_hot = ['0'] * len(classes)
                    one_hot[classes.index(chunk.label)] = '1'
                    if success:
                        print(f'{name.name},' + ','.join(one_hot) ,file=rf)
                        c_count += 1
                #print(f"{str(cdf.name)}: {c_count} of {len(chunk_definitions)} chunks created, {not_annot} ignored as not '{annot_name}'", file=rf)
            except Exception as ex:
                print(f"Error while processing {str(cdf.name)}: {ex}", file=sys.stderr)

            finally:
                df_processed += 1

        print(f"\n End of processing, {df_processed} of {len(chunk_def_files)} input files processed")

# sox : offset + duration   normalised to 3s, new offset + new duration
# flac, --tag=offest_to_call

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cmd_args = parse_commandline()
    print(cmd_args)
    do_the_butchery(cmd_args)
