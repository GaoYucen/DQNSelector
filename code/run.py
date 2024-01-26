from DegGreedy import resultDegGreedy
from CovGreedy import resultCovGreedy
from KTVoting import resultKTVoting
from FastSelector import resultFastSelector
from CELF import resultCELF

from config import get_config
params, _ = get_config()

n_subarea = 100
n_users = params.user_no
n_seed_list = [params.seed_no]
n_samplefile = 10

input_file_path = r'../dataset/data_1'

output_result_file_prefix = r'../Result/Result_DegGreedy/Result_DegGreedy_sample1_allEC_3000'
resultDegGreedy(n_subarea, n_users, n_samplefile, input_file_path, output_result_file_prefix)
output_result_file_prefix = '../Result/Result_CovGreedy/Result_CovGreedy_sample1allEC_3000'
resultCovGreedy(n_subarea, n_users, n_samplefile, input_file_path, output_result_file_prefix)
output_result_file_prefix = '../Result/Result_KTVoting/Result_KTVoting_3000'
resultKTVoting(n_subarea, n_users, n_samplefile,input_file_path, output_result_file_prefix)
output_result_file_prefix = '../Result/Result_FastSelector/Result_FastSelector_sample1allEC_3000'
resultFastSelector(n_subarea, n_users, n_samplefile, input_file_path, output_result_file_prefix)
output_result_file_prefix = '../Result/Result_CELF/Result_CELF_sample1_allEC_3000'
resultCELF(n_subarea, n_users, n_samplefile, input_file_path, output_result_file_prefix)