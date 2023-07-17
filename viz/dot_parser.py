def is_subgraph_decl(line: str) -> bool:
    tokens = line.strip().split(" ")
    return tokens[0] == "subgraph"


def find_outside_quotes(
    txt: str, find: str, start: int | None = None, end: int | None = None
) -> int:
    start_idx: int = 0 if start is None else start
    end_idx: int = len(txt) if end is None else end
    in_quotes: bool = False
    for i, char in enumerate(txt[start_idx:end_idx]):
        if not in_quotes and txt[start_idx + i].startswith(find):
            return start_idx + i
        if char == '"':
            in_quotes = not in_quotes
    return -1


def has_attribute_list(line: str) -> tuple[int, int] | None:
    open_bracket = find_outside_quotes(line, "[")
    if open_bracket == -1:
        return None
    close_bracket = find_outside_quotes(line, "]", open_bracket)
    if close_bracket == -1:
        return None
    return open_bracket, close_bracket


def is_node(line: str) -> bool:
    if (indices := has_attribute_list(line)) is not None:
        before_attr = line[: indices[0]]
        return not ("->" in before_attr or before_attr.strip() == "node")
    return False


def is_edge(line: str) -> bool:
    if (indices := has_attribute_list(line)) is not None:
        return "->" in line[: indices[0]]
    return False


def get_edge_endpoints(line: str) -> tuple[str, str] | None:
    if not is_edge(line):
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

    return src, dst


def get_attributes(line: str) -> dict[str, str]:
    # Isolate attributes from the rest of the line
    og_line = line
    indices = has_attribute_list(line)
    if indices is None:
        return {}
    line = line[indices[0] + 1 : indices[1]]

    # Parse all attributes using cursed logic
    all_attributes: dict[str, str] = {}
    while len(line) > 0:
        # Parse name
        eq_idx: int = find_outside_quotes(line, "=")
        if eq_idx == -1:
            break
        attr_name = line[:eq_idx].strip()
        line = line[eq_idx + 1 :]

        # Parse value
        attr_value: str | None = None
        for i in range(len(line)):
            if line[i] == '"':
                second_quote_idx: int = line[i + 1 :].find('"')
                if second_quote_idx == -1:
                    raise DOTParsingError(
                        f'Failed to find closing quote for value of "{attr_name}" '
                        f'attribute"',
                        og_line,
                    )
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
        if attr_value is None:
            raise DOTParsingError(
                f'Failed to parse value for attribute "{attr_name}"', og_line
            )

        # Add the attribute
        all_attributes[attr_name] = attr_value

        # Eat up the space to the next alphanumeric character, if any
        for i in range(len(line)):
            if str(line[i]).isalnum():
                line = line[i:]
                break

    return all_attributes


class DOTParsingError(Exception):
    def __init__(self, msg: str, line: str) -> None:
        super().__init__(f"{msg}\n\tIn: {line}")
