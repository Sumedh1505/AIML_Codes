

data_analysis(image_path, train_csv_path, trained_model_path, label_map_path,train_results_path)
data_analysis(image_path, valid_csv_path, trained_model_path, label_map_path,valid_results_path)
data_analysis(image_path, test_csv_path, trained_model_path, label_map_path,test_results_path)

merge_prediction(train_results_path,analysis_path,'train')
merge_prediction(valid_results_path,analysis_path,'valid')
merge_prediction(test_results_path,analysis_path,'test')


''' usage of  get_df_from_pdf '''
sys.path.append('D:/PROJECTS_ROOT/Codes/spivision/')
import get_df_from_pdf
dir_to_save_df= 'D:/PROJECTS_ROOT/DataServices/zillow/Office_documents_OCRed/'
base_dir= 'D:/PROJECTS_ROOT/DataServices/zillow/Office_documents_OCRed/'

check=get_df_from_pdf(base_dir,4.16)
alldf_multi,alldf= check.create_df()
alldf_multi.to_csv(dir_to_save_df+ 'alldf_jpgcoords.csv', index= False)
alldf.to_excel(dir_to_save_df+ 'alldf_pdfcoords.xlsx', index= False)
 