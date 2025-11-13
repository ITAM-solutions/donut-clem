"""
File name: text_format
Author: Fran Moreno
Last Updated: 9/2/2025
Version: 1.0
Description: TOFILL
"""
from docxtpl import RichText as R

import synthetic_generator.word_bank as word_bank
from synthetic_generator.utils.config import DocxConfig


def as_rich(text: str, config: DocxConfig, size) -> R:
    return R(text, font=config.font_family, size=size, color=config.font_color)


def to_rich(data: dict, config):
    content = data['formatted']['content']
    flags = data['formatted']['flags']

    type_ = data['type']

    if type_ == 'kv':
        if not flags:
            return rich('\n'.join(content), config)
        else:
            if "bold_keys" in flags or "bold_vals" in flags:
                text = R()
                for line in content:
                    key, value = line.split(':')
                    text = rich_add(text, f"{key}: ", config, bold=("bold_keys" in flags))
                    text = rich_add(text, f"{value}\n", config, bold=("bold_vals" in flags))
                return text
    elif type_ == 'kv_as_table':
        # return content
        table_content = []
        for line in content:
            key, value = line.split(':')
            table_content.append({
                'k': rich(key, config, bold=("bold_keys" in flags)),
                'v': rich(value, config, bold=("bold_values" in flags))
            })
        return table_content
    elif type_ == 'table':
        num_cols = len(data['raw']['labels'])
        font_size = word_bank.TABLE_FONT_SIZES.get(num_cols)
        for row in content:
            for idx, cell in enumerate(row['fields']):
                row['fields'][idx] = rich(cell, config, size=font_size)
        return content
    elif type_ == 'table_header':
        num_cols = len(data['raw']['labels'])
        font_size = word_bank.TABLE_FONT_SIZES.get(num_cols)
        content = [rich(header, config, bold=True, size=font_size) for header in content]
        return content
    else:
        return rich(content, config)


def rich(content, config, **kwargs):
    size = config.font_size if "size" not in kwargs else kwargs["size"]

    try:
        return R(content, font=config.font_family, size=size, color=config.font_color, **kwargs)
    except TypeError:
        return R(content, font=config.font_family, size=size, color=config.font_color,)


def rich_add(r, content, config, **kwargs):
    size = config.font_size if "size" not in kwargs else kwargs["size"]

    try:
        r.add(content, font=config.font_family, size=size, color=config.font_color, **kwargs)
    except TypeError:
        r.add(content, font=config.font_family, size=size, color=config.font_color,)
    finally:
        return r