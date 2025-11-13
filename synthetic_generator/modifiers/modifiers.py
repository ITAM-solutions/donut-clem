import numpy as np

from synthetic_generator.utils.text_format import to_rich


def to_text_lines(data: dict):
    data_format = data.get('type', 'text')
    raw_data = data.get('raw', {})

    if not raw_data:
        return {"content": [], "flags": []}
    if data_format == 'kv':
        formatted = [f"{k}:{v}" for k, v in raw_data.items()]
    elif data_format == 'table':
        formatted = raw_data
    elif data_format  in ["text_line", "text_long"]:
        formatted = raw_data
    elif data_format == "text_multiline":
        formatted = [f"{line}\n" for line in raw_data]
    else:
        formatted = raw_data

    return {
        "content": formatted,
        "flags": [],
    }


# def mod_to_rich_text(data: dict, config):
#     data_format = data.get('type', 'text')
#     raw_data = data.get('raw', {})
#
#     match data_format:
#         case 'kv':
#             return _kv_to_rich_text(raw_data, config)
#         case 'table':
#             return _table_to_rich_text(raw_data, config)
#         case _:
#             return _text_to_rich_text(raw_data, config)


def _kv_to_rich_text(data: dict, config):
    return to_rich([f'{k}:{v}' for k, v in data.items()], config)


def _table_to_rich_text(data: dict, config):
    return []


def _text_to_rich_text(data: dict, config):
    return []


def mod_align_left(data: dict, config):
    """
    Input:
    {
        'key1': 'val1',
        'key_long2': 'val2',
        'key_very_long3: 'value3',
    }

    Output: (in Rich format):
    "
    key1           :  val1
    key_long2      :  val2
    key_very_long3 :  value3
    "
    :param data:
    :param config:
    :return:
    """
    if data.get("type") != 'kv':
        return data

    raw = data.get("raw", {})
    if not raw:
        return data

    max_key_length = np.max([len(k) for k in raw.keys()])
    tab_char = '\t'
    lines = [f'{k + tab_char * (1 + (max_key_length - len(k)) // 2)}:  {v}' for k, v in raw.items()]
    # lines = [f'{k + '\t' * 3} :  {v}' for k, v in raw.items()]

    data['formatted']['content'] = lines
    return data


def mod_bold_keys(data: dict, config):
    if data.get("type") != 'kv':
        return data

    raw = data.get("raw", {})
    if not raw:
        return data

    data['formatted']['flags'] += ['bold_keys']
    return data


# def mod_as_table(data: dict, config):
#     if data.get("type", '') == "kv":
#         data['type'] = 'table'
#         return data


# def mod_get_labels(table_data: dict, config):
#     if table_data.get("type") != 'table':
#         return table_data
#
#     content = table_data['formatted']['content']
#     table_data['formatted']['content'] = content['labels']
#     table_data["type"] = "table_header"
#     return table_data
#
#
# def mod_get_items(table_data: dict, config):
#     if table_data.get("type") != 'table':
#         return table_data
#
#     content = table_data['formatted']['content']
#     table_data['formatted']['content'] = content['items']
#     return table_data

