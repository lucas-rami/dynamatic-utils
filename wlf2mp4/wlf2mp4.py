import sys
import os
import subprocess
import glob
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

VALID_WIRE: str = "_validArray_"
READY_WIRE: str = "_readyArray_"
COLOR_ATTR: str = "color"


class WireState(Enum):
    LOGIC_0 = 0
    LOGIC_1 = 1
    UNDEFINED = 2

    @staticmethod
    def from_log(token: str) -> "WireState":
        val: str = token[2:-3]
        if val == "1":
            return WireState.LOGIC_1
        elif val == "0":
            return WireState.LOGIC_0
        return WireState.UNDEFINED


@dataclass(frozen=True)
class Port:
    component: str
    port_id: int
    is_input: bool

    @staticmethod
    def from_full_name(full_name: str) -> "Port":
        # Take off the duv/ prefix
        full_name = full_name[4:]

        # Derive port ID
        port_id: int = int(full_name[full_name.rfind("_") + 1 :])

        if (valid_idx := full_name.find(VALID_WIRE)) != -1:
            return Port(full_name[:valid_idx], port_id, False)

        ready_idx = full_name.find(READY_WIRE)
        assert ready_idx != -1
        return Port(full_name[:ready_idx], port_id, True)


@dataclass
class Channel:
    src: Port
    dst: Port
    attributes: dict[str, str]
    src_dot_name: str
    dst_dot_name: str

    def __str__(self) -> str:
        attr: list[str] = [f'{name}="{val}"' for (name, val) in self.attributes.items()]
        return (
            f"\"{self.src_dot_name}\" -> \"{self.dst_dot_name}\" [{' '.join(attr)}]\n"
        )


class DOTGraph:
    name: str
    content: list[str]
    channels: list[Channel]
    port_to_channel: dict[Port, Channel]

    def __init__(self, filepath: str) -> None:
        self.name = Path(filepath).stem
        self.content = []
        self.channels = []
        self.port_to_channel = {}


def gen_log_file(wlf_file: str, out_file: str) -> str:
    # Produce list of objects in WLF
    wlf_name: str = Path(wlf_file).stem
    obj_lst_file: str = os.path.join(out_file, f"{wlf_name}_objects.lst")
    subprocess.run(f"wlfman items -v {wlf_file} > {obj_lst_file}", shell=True)

    # Filter the list of objects to only include valid and ready signals
    obj_filter_lst_file: str = os.path.join(out_file, f"{wlf_name}_objects_filter.lst")
    with open(obj_lst_file, "r") as obj_lst_handle:
        with open(obj_filter_lst_file, "w") as obj_filter_lst_handle:
            while line := obj_lst_handle.readline():
                # Only keep output valid/ready signals
                if VALID_WIRE in line or READY_WIRE in line:
                    obj_filter_lst_handle.write(line)

    # Produce filtered WLF file
    wlf_filter_file: str = os.path.join(out_file, f"{wlf_name}_filter.wlf")
    subprocess.run(
        f"wlfman filter -f {obj_filter_lst_file} -o {wlf_filter_file} {wlf_file}",
        shell=True,
    )

    # Produce log file
    log_file: str = os.path.join(out_file, f"{wlf_name}.log")
    subprocess.run(f"wlf2log -l duv {wlf_filter_file} > {log_file}", shell=True)

    return log_file


# Converts the .log file containing the signal changes to a CSV
def log2csv(graph: DOTGraph, log_file: str, out_file: str) -> None:
    id_to_signal: dict[int, Port] = {}
    cycle: int = 0
    state: dict[Port, WireState] = {}

    with open(os.path.join(out_file, "sim.csv"), "w") as sim:
        # Write the column names
        sim.write(f"cycle, src_component, src_port, dst_component, dst_port, state\n")

        # For each edge, initialize its first state as undefined
        for channel in graph.channels:
            # sim.write(
            #     f"{cycle}, {channel.src.component}, {channel.src.port_id}, "
            #     f"{channel.dst.component}, {channel.dst.port_id}, undefined\n"
            # )
            state[channel.src] = WireState.UNDEFINED
            state[channel.dst] = WireState.UNDEFINED

        sim.write("0,0,0,0,0\n")

        # Parse log file
        with open(log_file, "r") as log_file_handle:
            while line := log_file_handle.readline():
                tokens: list[str] = line.split(" ")
                if len(tokens) == 0:
                    break
                if tokens[0] == "D":
                    # This defines a signal to id mapping
                    port = Port.from_full_name(tokens[1])
                    id_to_signal[int(tokens[2])] = port
                elif tokens[0] == "T":
                    # This starts a new cycle (not necessarily consecutive)
                    time = int(tokens[1].replace(".", "")[:-1]) - 2000
                    if time < 0:
                        cycle = 0
                    else:
                        cycle = (time // 4000) + 1
                    pass
                elif tokens[0] == "S":
                    # This sets a signal to a specific value
                    port: Port = id_to_signal[int(tokens[1])]
                    state[port] = WireState.from_log(tokens[2])

                    # Retrieve the channel that the port is a part of
                    if port not in graph.port_to_channel:
                        continue
                    channel: Channel = graph.port_to_channel[port]
                    valid: WireState = state[channel.src]
                    ready: WireState = state[channel.dst]

                    channel_state: str
                    if valid != WireState.LOGIC_1 and ready == WireState.LOGIC_1:
                        channel_state = "ready"
                    elif valid == WireState.LOGIC_1 and ready != WireState.LOGIC_1:
                        channel_state = "valid"
                    elif valid == WireState.LOGIC_1 and ready == WireState.LOGIC_1:
                        channel_state = "valid+ready"
                    elif valid == WireState.LOGIC_0 and ready == WireState.LOGIC_0:
                        channel_state = "empty"
                    else:
                        channel_state = "undefined"

                    sim.write(
                        f"{cycle}, {channel.src.component}, {channel.src.port_id}, "
                        f"{channel.dst.component}, {channel.dst.port_id}, "
                        f"{channel_state}\n"
                    )


def dot_is_edge(line: str) -> bool:
    tokens = line.strip().split(" ")
    return len(tokens) >= 3 and tokens[1] == "->"


def dot_get_edge_endpoints(line: str) -> tuple[str, str, str, str] | None:
    if not dot_is_edge(line):
        return None

    # Extract source and destination endpoints
    tokens = line.strip().split(" ")
    src: str = tokens[0]
    dst: str = tokens[2]

    # Remove potential quotes around endpoints
    if src.startswith('"') and src.endswith('"'):
        src = src[1:-1]
    if dst.startswith('"') and dst.endswith('"'):
        dst = dst[1:-1]

    # Handle special case where name starts with _
    src_fixed = src[1:] if src.startswith("_") else src
    dst_fixed = dst[1:] if dst.startswith("_") else dst

    return src_fixed, src, dst_fixed, dst


def dot_get_attributes(line: str) -> dict[str, str]:
    # Isolate attributes from the rest of the line
    open_bracket = line.find("[")
    close_bracket = line.find("]")
    if open_bracket == -1 or close_bracket == -1:
        return {}
    line = line[open_bracket + 1 : close_bracket]

    # Parse all attributes using cursed logic
    all_attributes: dict[str, str] = {}
    while len(line) > 0:
        # Parse name
        eq_idx: int = line.find("=")
        assert eq_idx != -1
        attr_name = line[:eq_idx].strip()
        line = line[eq_idx + 1 :]

        # Parse value
        attr_value: str | None = None
        for i in range(len(line)):
            if line[i] == '"':
                second_quote_idx: int = line[i + 1 :].find('"')
                assert second_quote_idx != -1
                attr_value = line[i + 1 : i + 1 + second_quote_idx]
                line = line[i + second_quote_idx + 2 :]
                break
            if str(line[i]).isalnum():
                # Find the first space, or the first comma, or the end of the line
                space_idx: int = line[i:].find(" ")
                comma_idx: int = line[i:].find(",")
                r_idx: int = len(line)
                if space_idx < comma_idx and space_idx != -1:
                    r_idx = space_idx
                elif comma_idx < space_idx and comma_idx != -1:
                    r_idx = comma_idx
                attr_value = line[i:r_idx]
                line = line[r_idx + 1 :]
                break
        assert attr_value is not None

        # Add the attribute
        all_attributes[attr_name] = attr_value

        # Eat up the space to the next alphanumeric character, if any
        for i in range(len(line)):
            if str(line[i]).isalnum():
                line = line[i:]
                break

    return all_attributes


# Read the original DOT into memory and cache the sequence of edges
def parse_dot(dot_file: str) -> DOTGraph:
    graph: DOTGraph = DOTGraph(dot_file)
    with open(dot_file, "r") as dot_file_handle:
        while line := dot_file_handle.readline():
            endpoints = dot_get_edge_endpoints(line)
            if not endpoints:
                # Append the line as is to the file content
                graph.content.append(line)
                continue

            # Decode the edge into signals
            attributes = dot_get_attributes(line)
            assert "from" in attributes and "to" in attributes
            src_cmp, src_og_name, dst_cmp, dst_og_name = endpoints
            src_port_id: int = int(attributes["from"][3:]) - 1
            dst_port_id: int = int(attributes["to"][2:]) - 1

            # Overwrite the arrow style
            if "arrowhead" in attributes:
                del attributes["arrowhead"]
            if "arrowtail" in attributes:
                del attributes["arrowtail"]
            if "dir" in attributes:
                del attributes["dir"]

            # Append the channel to the list of channels
            src_port = Port(src_cmp, src_port_id, False)
            dst_port = Port(dst_cmp, dst_port_id, True)
            channel = Channel(src_port, dst_port, attributes, src_og_name, dst_og_name)
            graph.channels.append(channel)

            # Map both of the channel's endpoints to the channel they belong to for
            # quick access
            graph.port_to_channel[src_port] = channel
            graph.port_to_channel[dst_port] = channel

            # Append the slightly modified line as is to the file content
            graph.content.append(str(channel))
    return graph


def print_dot(graph: DOTGraph, state: dict[Port, WireState], out_dot_file: str):
    edge_idx: int = 0
    with open(out_dot_file, "w") as out_dot_file_handle:
        for dot_line in graph.content:
            if dot_is_edge(dot_line):
                # Lookup the state of the signals corresponding to the edge (valid
                # signal from the source, ready signal from the destination)
                info = graph.channels[edge_idx]
                edge_idx += 1
                valid: WireState = state.get(info.src, WireState.UNDEFINED)
                ready: WireState = state.get(info.dst, WireState.UNDEFINED)

                # Overwrite the color attribute
                color: str = "black"
                if valid == WireState.UNDEFINED or ready == WireState.UNDEFINED:
                    color = "black"
                elif valid == WireState.LOGIC_1 and ready == WireState.LOGIC_1:
                    color = "green"
                elif valid == WireState.LOGIC_1:
                    color = "red"
                elif ready == WireState.LOGIC_1:
                    color = "blue"
                else:
                    color = "grey"
                info.attributes[COLOR_ATTR] = color

                out_dot_file_handle.write(str(info))
            else:
                # Write lines that are not edges as is
                out_dot_file_handle.write(dot_line)


def gen_dots(graph: DOTGraph, log_file: str, out_path: str, n_phases: int = -1):
    # Create output directory
    dot_idx: int = 0
    out_dir: str = os.path.join(out_path, "dots")
    out_dot_name: str = os.path.join(out_dir, graph.name)
    subprocess.run(f"mkdir -p {out_dir}", shell=True)

    id_to_signal: dict[int, Port] = {}
    state: dict[Port, WireState] = {}

    # Parse log file
    with open(log_file, "r") as log_file_handle:
        while line := log_file_handle.readline():
            tokens: list[str] = line.split(" ")
            if len(tokens) == 0:
                break
            if tokens[0] == "D":
                # This defines a signal to id mapping
                endpoint = Port.from_full_name(tokens[1])
                id_to_signal[int(tokens[2])] = endpoint
                state[endpoint] = WireState.from_log(tokens[3])
            elif tokens[0] == "T":
                # This starts a new phase
                idx: str = str(dot_idx).zfill(5)
                print_dot(graph, state, f"{out_dot_name}_{idx}.dot")
                dot_idx += 1
                if dot_idx == n_phases:
                    return
            elif tokens[0] == "S":
                # This sets a signal to a specific value
                state[id_to_signal[int(tokens[1])]] = WireState.from_log(tokens[2])


def convert_to_png(og_dot_file: str, out_path: str):
    # Create outpur directory
    og_dot_name: str = Path(og_dot_file).stem
    out_dir: str = os.path.join(out_path, "images")
    subprocess.run(f"mkdir -p {out_dir}", shell=True)

    # Read all DOTs from directory
    out_dot_name: str = os.path.join(out_path, "dots", og_dot_name)
    all_dot_files: list[str] = sorted(glob.glob(f"{out_dot_name}_*.dot"))
    for dot_file in all_dot_files:
        dot_name: str = Path(dot_file).stem
        png_file: str = os.path.join(out_dir, dot_name)
        subprocess.run(f"dot -Tpng {dot_file} > {png_file}.png", shell=True)


def create_video(dot_file: str, out_path: str):
    dot_name: str = Path(dot_file).stem
    image_dir: str = os.path.join(out_path, "images", dot_name)
    subprocess.run(
        f"ffmpeg -r 2 -i {image_dir}_%05d.png -vcodec libx264 "
        f'-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -y -an {out_path}/video.mp4',
        shell=True,
    )


def wlf2mp4() -> None:
    # We need a path to a DOT and to a WLF, as well as an output path
    if len(sys.argv) != 4:
        raise Exception("Script needs 3 arguments")

    dot_file: str = sys.argv[1]
    wlf_file: str = sys.argv[2]
    out_path: str = sys.argv[3]

    # Delete output folder if it exists, then recreate it
    subprocess.run(f"rm -rf {out_path}", shell=True)
    subprocess.run(f"mkdir -p {out_path}", shell=True)

    log_file: str = gen_log_file(wlf_file, out_path)
    graph: DOTGraph = parse_dot(dot_file)
    print("Converting to CSV...")
    log2csv(graph, log_file, out_path)
    print("Generating DOTs...")
    gen_dots(graph, log_file, out_path, n_phases=100)
    print("Converting to PNGs...")
    convert_to_png(dot_file, out_path)
    print("Creating video...")
    create_video(dot_file, out_path)


if __name__ == "__main__":
    wlf2mp4()
