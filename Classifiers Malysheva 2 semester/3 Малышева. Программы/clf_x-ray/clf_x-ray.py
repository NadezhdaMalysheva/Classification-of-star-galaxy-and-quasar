"""
Small module containing functionality to make predict for сlassification of objects into stars, quasars and galaxy
"""
import argparse
import json
import joblib
import functools
import glob
import importlib
import os
import multiprocessing
import pickle
import re
import shutil
#import subprocess
import sys
import time
from collections import defaultdict

import astropy.table
import numpy as np
import pandas as pd
import tqdm
import warnings

def _import_user_defined_features_transformation(
        path_to_code: str,
        transformation_name: str):
    path, module_name = os.path.split(os.path.splitext(path_to_code)[0])
    sys.path.append(path)
    user_module = importlib.import_module(module_name)
    return getattr(user_module, transformation_name)


def _columns_intersection(df1: pd.DataFrame, df2: pd.DataFrame):
    result = list()
    for col in df1.columns:
        if col in df2.columns:
            result.append(col)

    return result


def format_message(msg):
    return f'===== {msg} ====='


def file_exists(path):
    return os.path.isfile(path) and os.stat(path).st_size > 0


def parse_cli_args():
    def check_args(args):
        return True

    def _add_default_and_type(desc: str, arg_type=None, default_value=None,
                              isflag=False):
        """
        Helper function to form an argument description for argparse
        :param desc: helper description
        :param arg_type: argument type to insert to argument description
        :param default_value: default value of arg_type to insert to argument description
        :param isflag: if True, then adds info that the argument is a flag
        :return: argument description with type and default value information if such provided
        """
        assert arg_type is None or callable(arg_type)
        assert default_value is None or isinstance(default_value, arg_type)

        if isflag:
            default_value_msg = ''
            arg_type_msg = 'flag'
        else:
            arg_type_msg = ''
            if arg_type is not None:
                arg_type_msg = f'type: {type(arg_type()).__name__}'

            default_value_msg = ''
            if default_value is not None:
                if arg_type == str:
                    default_value_msg = f'default: "{default_value}"'
                else:
                    default_value_msg = f'default: {default_value}'

        if default_value_msg and arg_type_msg:
            return f'[{arg_type_msg}; {default_value_msg}] {desc}'

        if arg_type_msg:
            return f'[{arg_type_msg}] {desc}'

        if default_value_msg:
            return f'[{default_value_msg}] {desc}'

        return desc

    description = "Script to make сlassification of objects into stars, quasars and galaxy for objects with photometric data."\
                  "\n" \
                  "\n List of models" \
                  "\n " \
                  "\n   - gb     - Gradient boosting models" \
                  "\n   - tn     - TabNet models" \
                  "\n   - gb_big - Gradient boosting models trained on a large dataset" \
                  "\n   - tn_big - TabNet models trained on a large dataset" \
                  "\n          - 18 - SDSS + WISE" \
                  "\n          - 19 - PanSTARRS + WISE" \
                  "\n          - 20 - SDSS + DESI LIS + WISE" \
                  "\n          - 21 - PanSTARRS + DESI LIS + WISE" \
                  "\n          - 22 - DESI LIS + WISE" \
                  "\n          - 34 - SDSS + PanSTARRS + WISE" \
                  "\n          - 35 - SDSS + PanSTARRS + DESI LIS + WISE"
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=description)


    argument_description = "Path to input file."
    arg_type = str
    default_value = './x-ray_data.gz_pkl'
    parser.add_argument('--inputFile', type=arg_type,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))
    
    argument_description = "Path to dir with features files to make predictions on."
    arg_type = str
    default_value = None
    parser.add_argument('--predictOn', type=arg_type, default=default_value,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    argument_description = "Format of output file."
    arg_type = str
    default_value = 'gz_pkl'
    parser.add_argument('--outputExt', type=arg_type,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    # other arguments
    argument_description = "Path to output directory."
    arg_type = str
    default_value = None
    parser.add_argument('-o', '--outputDir', type=arg_type,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    argument_description = "Models series to apply. Possible values are 'gb', 'tn'."
    arg_type = str
    default_value = "gb"
    parser.add_argument('--modelsSeries', type=arg_type, default=default_value,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    argument_description = "Specify list of models' ids to apply. If not specified, then will apply default" \
                           " set of models for specified series. See list of models above."
    arg_type = int
    default_value = None
    parser.add_argument('--modelsIds', type=arg_type, default=default_value,
                        nargs='+',
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    argument_description = "If used then all models will be loaded into memory at once" 
    parser.add_argument('--keepModelsInMemory', action='store_true',
                        help=_add_default_and_type(argument_description,
                                                   isflag=True))

    argument_description = "Predictions chunk size. Ignored if --predictOn"
    arg_type = int
    default_value = 100000
    parser.add_argument('--chunkSize', type=arg_type, default=default_value,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    argument_description = "Number of jobs for parallelism"
    arg_type = int
    default_value = 1
    parser.add_argument('--njobs', type=arg_type, default=default_value,
                        help=_add_default_and_type(argument_description,
                                                   arg_type, default_value))

    return parser.parse_args()


def _drop_multidims(table: astropy.table.Table):
        """
        drop multidimentional columns from astropy.Table so it can be converted to pandas.DataFrame
        """
        singledim_cols = list()
        multidim_cols = list()
        for col in table.colnames:
            if len(table[col].shape) == 1:
                singledim_cols.append(col)
            else:
                multidim_cols.append(col)

        return table[singledim_cols], multidim_cols

def read_table(table):

        if isinstance(table, str):
            _, ext = os.path.splitext(table)

            if ext == '.gz_pkl':
                try:
                    return pd.read_pickle(table, compression='gzip')
                except:
                    return pd.read_pickle(table)

            if ext == '.pkl':
                return pd.read_pickle(table)

            if ext == '.fits':
                table = astropy.table.Table.read(table)

        if isinstance(table, pd.DataFrame):
            return table

        if isinstance(table, astropy.table.Table):
            table, dropped_cols = _drop_multidims(table)
            if dropped_cols:
                warnings.warn(
                    "multidimentional columns are dropped from table : {}".format(
                        dropped_cols))

            return table.to_pandas()

        raise Exception('Unsupported format of table')

def write_table(table, df):
        if not isinstance(df, pd.DataFrame):
            raise Exception('DataFrame is not pandas')

        if isinstance(table, str):
            _, ext = os.path.splitext(table)

            if ext == '.gz_pkl':
                df.to_pickle(table, compression='gzip', protocol=4)
                return

            if ext == '.pkl':
                df.to_pickle(table)
                return

            if ext == '.fits':
                #print(df.fillna(-9999).dtypes)
                df['index'] = df.index.copy()
                df = astropy.table.Table.from_pandas(df.fillna(-9999))#.dropna()

                '''
                df, dropped_cols = _drop_multidims(df)
                if dropped_cols:
                        warnings.warn(
                            "multidimentional columns are dropped from table : {}".format(
                                dropped_cols))
                '''
                df.write(table, overwrite=True, format='fits')
                return

        raise Exception('Unsupported format of table')


def _load_obj(*args):
    return joblib.load(os.path.join(*args))

def pred(data, feats, model, mid, robust=None):
    
    X = data[feats].values
    if robust is not None:
        X = robust.transform(X)
    y_t = model.predict(X)
    y_p = model.predict_proba(X)
    p = pd.DataFrame({
                                #index:                data[index],
                                f'ProbabilityS{mid}': y_p[:, 0],
                                f'ProbabilityQ{mid}': y_p[:, 1],
                                f'ProbabilityG{mid}': y_p[:, 2],
                                f'Label{mid}':        y_t
                        }, index=data['__tempid__'])

    #p.index('__tempid__')
    return p



def predict(datasets_files, models_path, models: dict, njobs=1,
                                 output_path='./', keep_in_memory=False):#, index='nrow'):

    models_data = defaultdict(dict)
    models_iterable = tqdm.tqdm(models.items(),
                                desc="Load models") if keep_in_memory else models.items()
    for mid, model in models_iterable:
        clf_path = os.path.join(models_path, f'model_{model}.pkl')
        features_path = os.path.join(models_path, f'features_{model}.pkl')
        models_data[mid]['clf'] = _load_obj(
            clf_path) if keep_in_memory else clf_path
        models_data[mid]['feats'] = _load_obj(
            features_path) if keep_in_memory else features_path
        if mid[:2]=='gb':
            robust_path = os.path.join(models_path, f'{model}_robust_for_gb.pkl')
            models_data[mid]['robust'] = _load_obj(
            robust_path) if keep_in_memory else robust_path
    

    for ds_path in tqdm.tqdm(datasets_files, desc="Predictions"):

        fname, ext = os.path.splitext(ds_path)
        fname = os.path.splitext(fname)[0]#.split('/')[-1]

        test_data = read_table(os.path.join(ds_path))
        #need_to_predict = index in test_data.columns
        #if not need_to_predict:
        #    continue

        test_data['__tempid__'] = test_data.index.copy()

        #############можно разбить на потоки

        for mid, model_data in tqdm.tqdm(models_data.items()):
            preds_dst_file = f'{fname}.preds.{mid}{ext}'#############################################
            if not file_exists(preds_dst_file):
                if keep_in_memory:
                    feats = model_data['feats']
                else:
                    feats = _load_obj(model_data['feats'])

                notna_mask = test_data[feats].notna().all(axis=1)
                if not notna_mask.any():
                    continue

                test_data_notna = test_data.loc[notna_mask]

                if keep_in_memory:
                    clf = model_data['clf']
                else:
                    clf = _load_obj(model_data['clf'])
                
                res = pd.DataFrame()

                if mid[:2]=='gb':
                    if keep_in_memory:
                        robust = model_data['robust']
                    else:
                        robust = _load_obj(model_data['robust'])
                    res = pred(test_data_notna, feats, clf, mid, robust)
                else:
                    res = pred(test_data_notna, feats, clf, mid)


                if not keep_in_memory:
                    del clf

                res.to_pickle(preds_dst_file, compression='gzip', protocol=4)


def split_data(data=None, chunksize=100000):
    
    for istart in range(0, len(data), chunksize):
        iend = min(istart + chunksize, len(data) + 1)
        result = data.iloc[istart:iend]
        yield result

def assemble_results(buf_path: str, dst_path: str, models_series, format_save: str):
    start_nrow = 0
    
    for file in sorted(glob.glob(os.path.join(buf_path, '*.features.gz_pkl'))):
        # shutil.copy(file, dst_path)
        fname = re.findall(r'(.*)\.features\.gz_pkl$', os.path.basename(file))[0]

        preds_dst_file = os.path.join(dst_path,
                                      f'{fname}.predictions.{models_series}.{format_save}')
        
        preds = []
        for preds_file in glob.glob(
                os.path.join(buf_path, f'{fname}.p*.{models_series}*')):
            #print(preds_file)
            preds.append(pd.read_pickle(preds_file, compression='gzip'))

        try:
            data = pd.concat(preds, axis=1)
        except ValueError as ve:
            if str(ve) == 'No objects to concatenate':
                continue
            else:
                raise ValueError(ve)

        write_table(preds_dst_file, data)


def main():
    
    data_path = './'
    models_path = os.path.join(data_path, 'models')

    models_series = {
        "gb": {
            "path": os.path.join(models_path, 'gb'),
            "models": {
                "18": "sdssdr16+wise_decals8tr",
                "19": "psdr2+wise_decals8tr",
                "20": "sdssdr16+all_decals8tr",
                "21": "psdr2+all_decals8tr",
                "22": "decals8tr",
                "34": "sdssdr16+psdr2+wise_decals8tr",
                "35": "sdssdr16+psdr2+all_decals8tr"
            }
        },
        "tn": {
            "path": os.path.join(models_path, 'tn'),
            "models": {
                "18": "sdssdr16+wise_decals8tr",
                "19": "psdr2+wise_decals8tr",
                "20": "sdssdr16+all_decals8tr",
                "21": "psdr2+all_decals8tr",
                "22": "decals8tr",
                "34": "sdssdr16+psdr2+wise_decals8tr",
                "35": "sdssdr16+psdr2+all_decals8tr"
            }
        },
        "gb_big": {
            "path": os.path.join(models_path, 'gb'),
            "models": {
                "18": "sdssdr16+wise_decals8tr",
                "19": "psdr2+wise_decals8tr",
                "20": "sdssdr16+all_decals8tr",
                "21": "psdr2+all_decals8tr",
                "22": "decals8tr",
                "34": "sdssdr16+psdr2+wise_decals8tr",
                "35": "sdssdr16+psdr2+all_decals8tr"
            }
        },
        "tn_big": {
            "path": os.path.join(models_path, 'tn'),
            "models": {
                "18": "sdssdr16+wise_decals8tr",
                "19": "psdr2+wise_decals8tr",
                "20": "sdssdr16+all_decals8tr",
                "21": "psdr2+all_decals8tr",
                "22": "decals8tr",
                "34": "sdssdr16+psdr2+wise_decals8tr",
                "35": "sdssdr16+psdr2+all_decals8tr"
            }
        }
    }
    files2predict = []
    args = parse_cli_args()
    #print('OOON', args.predictOn)
    if args.inputFile is None :
                inputFile = './x-ray_data.gz_pkl'
    else:
                inputFile = args.inputFile
    if args.outputDir is None :
                outputDir = './'
    else:
                outputDir = args.outputDir
    if args.predictOn is None:
        
        files2predict = []
        input_data_file = read_table(inputFile)
        data_path = os.path.join(outputDir, 'data')
        #print('dataaaaaaaaaaaa', data_path)
        data_written_file = os.path.join(data_path, "DATA_WRITTEN_FILE.txt")
        if not os.path.isfile(data_written_file):
                os.makedirs(data_path, exist_ok=True)
                iterator = list(split_data(data = input_data_file,
                                            chunksize=args.chunkSize))
                

                for i, chunk in tqdm.tqdm(enumerate(iterator), total=len(iterator),
                                        desc='Preparing data'):
                    fname = 'part-{:05d}'.format(i)
                    chunk_dst_path = os.path.join(data_path,
                                                    f'{fname}.features.gz_pkl')
                    chunk.to_pickle(chunk_dst_path, compression='gzip',
                                            protocol=4)
        predictOn = data_path
    else:
        predictOn = args.predictOn
        
    buf_path = os.path.join(outputDir, 'buf')
    os.makedirs(buf_path, exist_ok=True)
    for file in glob.glob(os.path.join(predictOn, '*.features.gz_pkl')):
        shutil.copy(file, buf_path)
        files2predict.append(os.path.join(buf_path, os.path.basename(file)))

    
	       	
    if args.modelsIds is not None:
        
            #print(models_series[args.modelsSeries])

            models_path = models_series[args.modelsSeries]['path']
            models = {f'{args.modelsSeries}{mid}': model for mid, model in
                      models_series[args.modelsSeries]['models'].items()
                      if int(mid) in args.modelsIds}

            files2predict = sorted(files2predict)
            #print(files2predict, models_path, models)

            predict(files2predict, models_path, models,
                    keep_in_memory=args.keepModelsInMemory, 
                    njobs=args.njobs)

            assemble_results(buf_path, args.outputDir,
                             models_series=args.modelsSeries,
                             format_save=args.outputExt if args.outputExt is not None else 'gz_pkl')
        # if args.cleanupBuffer:
        #     shutil.rmtree(buf_path)


if __name__ == '__main__':
    main()
