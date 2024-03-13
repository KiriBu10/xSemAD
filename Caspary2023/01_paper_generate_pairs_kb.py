from sad.utils.log_to_pair import LogToPairDictionaryTransformer
from pathlib import Path
#from sad.utils.input_handler import pairs_dir, remove_files, models_dir
data_type='train'
models_dir=f'input/sap_sam_2022/filtered/{data_type}/logs/'
pairs_dir = f'input/sap_sam_2022/filtered/{data_type}/pairs/'
INCLUDE_TRACES_WITH_LOOPS = True
SPLIT_LOOPS_IN_SUBTRACES = False
REMOVE_NON_ENGLISH_LABELS = True

log_to_pair_dictionary_transformer = LogToPairDictionaryTransformer(
   models_dir, None, pairs_dir # log_dir, transformer_dir, svm_dir
)
log_to_pair_dictionary_transformer.log_to_dictionary_record(
   INCLUDE_TRACES_WITH_LOOPS,
   SPLIT_LOOPS_IN_SUBTRACES,
   REMOVE_NON_ENGLISH_LABELS,
   False,
   False,
)