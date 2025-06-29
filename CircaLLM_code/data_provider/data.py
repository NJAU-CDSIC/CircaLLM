import numpy as np
import math
import copy
def load_from_tsfile(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    return_meta_data=False,
    return_type="auto",
    Realcase=False
):
    """Load time series .ts file into X and (optionally) y.

    Parameters
    ----------
    full_file_path_and_name : string
        full path of the file to load, .ts extension is assumed.
    replace_missing_vals_with : string, default="NaN"
        issing values in the file are replaces with this value
    return_meta_data : boolean, default=False
        return a dictionary with the meta data loaded from the file
    return_type : string, default = "auto"
        data type to convert to.
        If "auto", returns numpy3D for equal length and list of numpy2D for unequal.
        If "numpy2D", will squash a univariate equal length into a numpy2D (n_cases,
        n_timepoints). Other options are available but not supported medium term.

    Returns
    -------
    data: Union[np.ndarray,list]
        time series data, np.ndarray (n_cases, n_channels, series_length) if equal
        length time series, list of [n_cases] np.ndarray (n_channels, n_timepoints)
        if unequal length series.
    y : target variable, np.ndarray of string or int
    meta_data : dict (optional).
        dictionary of characteristics, with keys
        "problemname" (string), booleans: "timestamps", "missing", "univariate",
        "equallength", "classlabel", "targetlabel" and "class_values": [],

    Raises
    ------
    IOError if the load fails.
    """
    # Check file ends in .ts, if not, insert
    if not full_file_path_and_name.endswith(".ts"):
        full_file_path_and_name = full_file_path_and_name + ".ts"
    # Open file
    with open(full_file_path_and_name, "r", encoding="utf-8") as file:
        # Read in headers
        meta_data = _load_header_info(file)
        # load into list of numpy
        data, y, meta_data = _load_data(file, meta_data,Realcase)
        data = np.mean(data, axis=1, keepdims=True)

    # if equal load to 3D numpy
    if meta_data["equallength"]:
        data = np.array(data)
        if return_type == "numpy2D" and meta_data["univariate"] and int(meta_data["dup"])==1:
            data = data.squeeze()#squeeze 方法用于去除数组中所有长度为1的维度
    # If regression problem, convert y to float
    if meta_data["targetlabel"]:
        y = y.astype(float)
    
    time_stamp=_get_timestamps_info(meta_data['timestamps'].copy(),size=data.shape[0]) if meta_data['timestamps'] else meta_data['timestamps']

    return (data, y, time_stamp, meta_data) if return_meta_data else (data, y, time_stamp)

def _get_timestamps_info(timestamps,size):
    time_stamp=np.array([[math.floor(t), (t - math.floor(t))*60] for t in timestamps])
    time_stamp = np.tile(time_stamp, (size, 1, 1))
    return time_stamp

def _load_header_info(file):
    """Load the meta data from a .ts file and advance file to the data.

    Parameters
    ----------
    file : stream.
        input file to read header from, assumed to be just opened

    Returns
    -------
    meta_data : dict.
        dictionary with the data characteristics stored in the header.
    """
    meta_data = {
        "problemname": "none",
        "timestamps": [],
        "missing": False,
        "univariate": True,
        "dup": 1,
        "equallength": True,
        "classlabel": True,
        "targetlabel": False,
        "class_values": [],
    }
    boolean_keys = ["missing", "univariate", "equallength", "targetlabel"]
    for line in file:
        line = line.strip().lower()
        if line and not line.startswith("#"):
            tokens = line.split(" ")
            token_len = len(tokens)
            key = tokens[0][1:]
            if key == "data":
                if line != "@data":
                    raise IOError("data tag should not have an associated value")
                return meta_data
            
            if key in meta_data.keys():
                if key in boolean_keys:
                    if token_len != 2:
                        raise IOError(f"{tokens[0]} tag requires a boolean value")
                    if tokens[1] == "true":
                        meta_data[key] = True
                    elif tokens[1] == "false":
                        meta_data[key] = False
                elif key == "problemname" or  key == "dup":
                    meta_data[key] = tokens[1]
                elif key == "timestamps":
                    meta_data[key] = [float(part) for part in tokens[1].split(',')] if tokens[1]!="false" else False
                elif key == "classlabel":
                    if tokens[1] == "true":
                        meta_data["classlabel"] = True
                        if token_len == 2:
                            raise IOError(
                                "if the classlabel tag is true then class values "
                                "must be supplied"
                            )
                    elif tokens[1] == "false":
                        meta_data["classlabel"] = False
                    else:
                        raise IOError("invalid class label value")
                    meta_data["class_values"] = [token.strip() for token in tokens[2:]]
        if meta_data["targetlabel"]:
            meta_data["classlabel"] = False
    return meta_data

def _load_data(file, meta_data,Realcase, replace_missing_vals_with="NaN"):
    """Load data from a file with no header.

    this assumes each time series has the same number of channels, but allows unequal
    length series between cases.

    Parameters
    ----------
    file : stream, input file to read data from, assume no comments or header info
    meta_data : dict.
        with meta data in the file header loaded with _load_header_info

    Returns
    -------
    data: list[np.ndarray].
        list of numpy arrays of floats: the time series
    y_values : np.ndarray.
        numpy array of strings: the class/target variable values
    meta_data :  dict.
        dictionary of characteristics enhanced with number of channels and series length
        "problemname" (string), booleans: "timestamps", "missing", "univariate",
        "equallength", "classlabel", "targetlabel" and "class_values": [],

    """
    data = []
    n_cases = 0
    n_channels = 0  # Assumed the same for all
    current_channels = 0
    series_length = 0
    y_values = []
    for line in file:
        line = line.strip().lower()
        line = line.replace("?", replace_missing_vals_with)
        channels = line.split(":")
        n_cases += 1
        current_channels = len(channels)
        if meta_data["classlabel"] or meta_data["targetlabel"]:
            current_channels -= 1
        if n_cases == 1:  # Find n_channels and length  from first if not unequal
            n_channels = current_channels
            if meta_data["equallength"]:
                series_length = len(channels[0].split(","))
        else:
            if current_channels != n_channels:
                raise IOError(
                    f"Inconsistent number of dimensions in case {n_cases}. "
                    f"Expecting {n_channels} but have read {current_channels}"
                )
            if meta_data["univariate"] and int(meta_data["dup"])==1:
                if current_channels > 1:
                    raise IOError(
                        f"Seen {current_channels} in case {n_cases}."
                        f"Expecting univariate from meta data"
                    )
        if meta_data["equallength"]:
            current_length = series_length
        else:
            current_length = len(channels[0].split(","))
        np_case = np.zeros(shape=(n_channels, current_length))#current_length是时间点的个数
        for i in range(0, n_channels):
            single_channel = channels[i].strip()
            data_series = single_channel.split(",")
            data_series = [float(x) for x in data_series]
            if len(data_series) != current_length:
                raise IOError(
                    f"Unequal length series, in case {n_cases} meta "
                    f"data specifies all equal {series_length} but saw "
                    f"{len(single_channel)}"
                )
            np_case[i] = np.array(data_series)
        data.append(np_case)
        if meta_data["classlabel"] or meta_data["targetlabel"]:
            temp_label=channels[n_channels]
            if Realcase:
                temp_label=[x for x in channels[n_channels].split(",")]
            y_values.append(temp_label)
    if meta_data["equallength"]:
        data = np.array(data)
    return data, np.asarray(y_values), meta_data#数据，标签，meta信息




def MultiGroup_load_from_tsfile(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    return_meta_data=False,
    return_type="auto",
    Realcase=False
):
    """Load time series .ts file into X and (optionally) y.

    Parameters
    ----------
    full_file_path_and_name : string
        full path of the file to load, .ts extension is assumed.
    replace_missing_vals_with : string, default="NaN"
        issing values in the file are replaces with this value
    return_meta_data : boolean, default=False
        return a dictionary with the meta data loaded from the file
    return_type : string, default = "auto"
        data type to convert to.
        If "auto", returns numpy3D for equal length and list of numpy2D for unequal.
        If "numpy2D", will squash a univariate equal length into a numpy2D (n_cases,
        n_timepoints). Other options are available but not supported medium term.

    Returns
    -------
    data: Union[np.ndarray,list]
        time series data, np.ndarray (n_cases, n_channels, series_length) if equal
        length time series, list of [n_cases] np.ndarray (n_channels, n_timepoints)
        if unequal length series.
    y : target variable, np.ndarray of string or int
    meta_data : dict (optional).
        dictionary of characteristics, with keys
        "problemname" (string), booleans: "timestamps", "missing", "univariate",
        "equallength", "classlabel", "targetlabel" and "class_values": [],

    Raises
    ------
    IOError if the load fails.
    """
    # Check file ends in .ts, if not, insert
    if not full_file_path_and_name.endswith(".ts"):
        full_file_path_and_name = full_file_path_and_name + ".ts"
    # Open file
    with open(full_file_path_and_name, "r", encoding="utf-8") as file:

        meta_data_1,meta_data_2 = _load_header_info_mulGroup(file)
        series_1,  meta_data_1,series_2,  meta_data_2 ,y= _load_data_mulGroup(file, meta_data_1,meta_data_2,Realcase)
        
        series_1, series_2 = np.mean(series_1, axis=1, keepdims=True),np.mean(series_2, axis=1, keepdims=True)
    # if equal load to 3D numpy
    if meta_data_1["equallength"]:
        data_1 = np.array(series_1)
        if return_type == "numpy2D" and meta_data_1["univariate"]:
            data_1 = data_1.squeeze()#squeeze 方法用于去除数组中所有长度为1的维度

    if meta_data_2["equallength"]:
        data_2 = np.array(series_2)
        if return_type == "numpy2D" and meta_data_2["univariate"]:
            data_2 = data_2.squeeze()#squeeze 方法用于去除数组中所有长度为1的维度

    # If regression problem, convert y to float
    if meta_data_1["targetlabel"] and meta_data_2["targetlabel"]:
        y = y.astype(float)    
    time_stamp_1=_get_timestamps_info_mulGroup(meta_data_1['timestamps'].copy(),size=data_1.shape[0]) if meta_data_1['timestamps'] else meta_data_1['timestamps']
    time_stamp_2=_get_timestamps_info_mulGroup(meta_data_2['timestamps'].copy(),size=data_2.shape[0]) if meta_data_2['timestamps'] else meta_data_2['timestamps']

    return (data_1, time_stamp_1, data_2, time_stamp_2,y,meta_data_1,meta_data_2) if return_meta_data else (data_1, time_stamp_1, data_2, time_stamp_2,y)

def _get_timestamps_info_mulGroup(timestamps,size):
    time_stamp=np.array([[math.floor(t), (t - math.floor(t))*60] for t in timestamps])
    time_stamp = np.tile(time_stamp, (size, 1, 1))
    return time_stamp

def _load_header_info_mulGroup(file):
    """Load the meta data from a .ts file and advance file to the data.

    Parameters
    ----------
    file : stream.
        input file to read header from, assumed to be just opened

    Returns
    -------
    meta_data : dict.
        dictionary with the data characteristics stored in the header.
    """
    meta_data_1 = {
        "problemname": "none",
        "timestamps": [],
        "missing": False,
        "univariate": True,
        "dup": 1,
        "equallength": True,
        "classlabel": True,
        "targetlabel": False,
        "class_values": [],
    }
    meta_data_2 = copy.deepcopy(meta_data_1)

    boolean_keys = ["missing", "univariate", "equallength", "targetlabel"]
    for line in file:
        line = line.strip().lower()
        if line and not line.startswith("#"):
            tokens = line.split(" ")
            token_len = len(tokens)
            key = tokens[0][1:]
            if key == "data":
                if line != "@data":
                    raise IOError("data tag should not have an associated value")
                break
            if key in meta_data_1.keys():
                if key in boolean_keys:
                    if token_len != 2:
                        raise IOError(f"{tokens[0]} tag requires a boolean value")
                    if tokens[1] == "true":
                        meta_data_1[key] = True
                        meta_data_2[key] = True
                    elif tokens[1] == "false":
                        meta_data_1[key] = False
                        meta_data_2[key] = False
                elif key == "problemname" or  key == "dup":
                    meta_data_1[key] = tokens[1] 
                    meta_data_2[key] = tokens[1]
                elif key == "timestamps":
                    if ";" in line:
                        temp_time_1,temp_time_2=tokens[1].split(';')[0],tokens[1].split(';')[1]
                        meta_data_1[key] = [float(part) for part in temp_time_1.split(',')] if temp_time_1!="false" else False    
                        meta_data_2[key] = [float(part) for part in temp_time_2.split(',')] if temp_time_2!="false" else False

                elif key == "classlabel":
                    if tokens[1] == "true":
                        meta_data_1["classlabel"],meta_data_2["classlabel"] = True,True
                        if token_len == 2:
                            raise IOError(
                                "if the classlabel tag is true then class values "
                                "must be supplied"
                            )
                    elif tokens[1] == "false":
                        meta_data_1["classlabel"],meta_data_2["classlabel"] = False,False
                    else:
                        raise IOError("invalid class label value")
                    meta_data_1["class_values"] = [token.strip() for token in tokens[2:]]
                    meta_data_2["class_values"] = [token.strip() for token in tokens[2:]]
        if meta_data_1["targetlabel"]:
            meta_data_1["classlabel"],meta_data_2["classlabel"] = False,False

    return meta_data_1,meta_data_2

def _load_data_mulGroup(file, meta_data_1,meta_data_2,Realcas,replace_missing_vals_with="NaN"):
    """Load data from a file with no header.

    this assumes each time series has the same number of channels, but allows unequal
    length series between cases.

    Parameters
    ----------
    file : stream, input file to read data from, assume no comments or header info
    meta_data : dict.
        with meta data in the file header loaded with _load_header_info

    Returns
    -------
    data: list[np.ndarray].
        list of numpy arrays of floats: the time series
    y_values : np.ndarray.
        numpy array of strings: the class/target variable values
    meta_data :  dict.
        dictionary of characteristics enhanced with number of channels and series length
        "problemname" (string), booleans: "timestamps", "missing", "univariate",
        "equallength", "classlabel", "targetlabel" and "class_values": [],

    """
    series_1,series_2=[],[]
    y_values=[]
    n_cases = 0
    n_channels_1,n_channels_2 = 0,0#有几个channel
    current_channels_1,current_channels_2 = 0,0#当前行有几个channel
    series_length_1,series_length_2 = 0,0
    # print(f'meta_data_1:{meta_data_1}')
    # print(f'meta_data_2:{meta_data_2}')
    for line in file:
        line = line.strip().lower()
        line = line.replace("?", replace_missing_vals_with)
        
        data_1,data_2,label = line.split(";")
        
        if Realcas:
            parts=label.split("_")
            first_part = parts[0]  # 第一部分是 "abcc9"
            numbers_part = parts[1]
            numbers = [int(num) for num in numbers_part.split(",")]
            label = [first_part] + numbers
        else:
            label=[x for x in label.split(",")]
        
        channels_1,channels_2 = data_1.split(":"),data_2.split(":")
        n_cases += 1
        current_channels_1,current_channels_2 = len(channels_1),len(channels_2)

        if n_cases == 1:  # Find n_channels and length  from first if not unequal
            n_channels_1,n_channels_2 = current_channels_1,current_channels_2
            if meta_data_1["equallength"]:
                series_length_1 = len(channels_1[0].split(","))
            if meta_data_2["equallength"]:
                series_length_2 = len(channels_2[0].split(","))
        else:
            if current_channels_1 != n_channels_1 or current_channels_2 != n_channels_2:
                raise IOError(
                    f"Inconsistent number of dimensions in case {n_cases}. "
                    f"Expecting {n_channels_1} but have read {current_channels_1}"
                    f"Expecting {n_channels_2} but have read {current_channels_2}"
                )
            
            if (meta_data_1["univariate"] and int(meta_data_1["dup"])==1) or (meta_data_2["univariate"] and int(meta_data_2["dup"])==1):
                if current_channels_1 > 1 or current_channels_2 > 1:
                    raise IOError(
                        f"Seen {current_channels_1} in case {n_cases}."
                        f"Seen {current_channels_2} in case {n_cases}."
                        f"Expecting univariate from meta data"
                    )
                
        if meta_data_1["equallength"]:
            current_length_1 = series_length_1
        else:
            current_length_1 = len(channels_1[0].split(","))

        if meta_data_2["equallength"]:
            current_length_2 = series_length_2
        else:
            current_length_2 = len(channels_2[0].split(","))
        np_case_1 = np.zeros(shape=(n_channels_1, current_length_1))#current_length是时间点的个数
        np_case_2 = np.zeros(shape=(n_channels_2, current_length_2))#current_length是时间点的个数

        for i in range(0, n_channels_1):
            single_channel = channels_1[i].strip()
            data_series = single_channel.split(",")
            data_series = [float(x) for x in data_series]
            if len(data_series) != current_length_1:
                raise IOError(
                    f"Unequal length series, in case {n_cases} meta "
                    f"data specifies all equal {series_length_1} but saw "
                    f"{len(single_channel)}"
                )
            np_case_1[i] = np.array(data_series)
        series_1.append(np_case_1)

        for i in range(0, n_channels_2):
            single_channel = channels_2[i].strip()
            data_series = single_channel.split(",")
            data_series = [float(x) for x in data_series]
            if len(data_series) != current_length_2:
                raise IOError(
                    f"Unequal length series, in case {n_cases} meta "
                    f"data specifies all equal {series_length_2} but saw "
                    f"{len(single_channel)}"
                )
            np_case_2[i] = np.array(data_series)
        series_2.append(np_case_2)

        if (meta_data_1["classlabel"] or meta_data_1["targetlabel"]) and (meta_data_2["classlabel"] or meta_data_2["targetlabel"]):
            y_values.append(label)
    if meta_data_1["equallength"]:
        data_1 = np.array(data_1)
    if meta_data_2["equallength"]:
        data_2 = np.array(data_2)
    return series_1, meta_data_1, series_2, meta_data_2, np.asarray(y_values)