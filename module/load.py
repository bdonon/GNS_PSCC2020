import pandas as pd
def _parse_line(line):
    """
    Do a regex search against all defined regexes and
    return the key and match result of the first matching regex

    """

    for key, rx in rx_dict.items():
        match = rx.search(line)
        if match:
            return key, match
    # if there are no matches
    return None, None

def int_or_float(s):
    try:
        return int(s)
    except:
        try:
            return float(s)
        except:
            return s

def parse_file(filepath):
    """
    Parse text at given filepath

    Parameters
    ----------
    filepath : str
        Filepath for file_object to be parsed

    Returns
    -------
    data : pd.DataFrame
        Parsed data

    """

    data = {}  # create an empty list to collect the data
    # open the file and read through it line by line
    with open(filepath, 'r') as file_object:
        line = file_object.readline()
        current_df = None
        while line:
            if line == '];\n':
                current_df = None
            elif line == 'mpc.bus = [\n':
                current_df = 'buses'
                columns = ['bus_i', 'type', 'Pd', 'Qd', 'Gs', 'Bs', 'area', 'Vm', 'Va', 'baseKV', 'zone', 'Vmax', 'Vmin']
                data[current_df] = pd.DataFrame(columns=columns)
                current_index = 0
            elif line == 'mpc.gen = [\n':
                current_df = 'gens'
                columns = ['bus', 'Pg', 'Qg', 'Qmax', 'Qmin', 'Vg', 'mBase', 'status', 'Pmax', 'Pmin', 'Pc1', 'Pc2', 'Qc1min', 'Qc1max', 'Qc2min', 'Qc2max', 'ramp_agc', 'ramp_10', 'ramp_30', 'ramp_q', 'apf']
                data[current_df] = pd.DataFrame(columns=columns)
                current_index = 0
            elif line == 'mpc.branch = [\n':
                current_df = 'lines'
                columns = ['fbus', 'tbus', 'r', 'x', 'b', 'rateA', 'rateB', 'rateC', 'ratio', 'angle', 'status', 'angmin', 'angmax']
                data[current_df] = pd.DataFrame(columns=columns)
                current_index = 0
            elif current_df is not None:
                line = line[:-2]
                data[current_df].loc[current_index] = [int_or_float(el) for el in line.split()]
                current_index += 1
            elif line[:11] == 'mpc.baseMVA':
                data['baseMVA'] = float([el for el in line.split()][2][:-1])
            else:
                line = file_object.readline()
                continue
            
            line = file_object.readline()
            #print(data[current_df])

    return data