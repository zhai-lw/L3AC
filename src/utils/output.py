MAX_NEST_LEVEL = 10


def dictionary(d: dict, out_fun=print, revert=False, ):
    out_str = f"\n{'-' * 64}\n"
    out_str += render_dict(d, revert=revert, prefix="| ")
    out_str += f"{'-' * 64}\n"
    out_fun(out_str)


def render_dict(d: dict, revert=False, prefix="", nest_level=0):
    out_str = ""
    for key, value in d.items():
        str_key = str(key)
        if not str_key.startswith('__'):
            if revert:
                out_str += f"{prefix}{value} :\t {str_key} \n"
            else:
                if isinstance(value, dict) and nest_level <= MAX_NEST_LEVEL:
                    out_str += f"{prefix}{str_key} :\n" \
                               f"{render_dict(value, revert, prefix=prefix + '| ', nest_level=nest_level + 1)}"
                else:
                    out_str += f"{prefix}{str_key} :\t {value} \n"
    return out_str


def variable(var, out_fun=print):
    dictionary(var.__dict__, out_fun=out_fun)
