# nova-spartha-macrophage-study
Repo with code for Macrophage phenotype detection methodology on textured surfaces via nuclear morphology using machine learning.
Includes preprocessing, training, and model evaluation. 
Notebook for initial data exploration and Cell Profiler pipelines, used to extract nuclei parameters.

Notebooks are available here for:

Data exploration:
https://colab.research.google.com/github/preste-ai/Macrophage-phenotype-classification/blob/main/Notebooks/Dim_reduction_analysis/tSNE_UMAP_nucleus_CD86_P4G4.ipynb
https://colab.research.google.com/github/preste-ai/Macrophage-phenotype-classification/blob/main/Notebooks/Dim_reduction_analysis/tSNE_UMAP_nucleus_CD86_TCPS.ipynb
https://colab.research.google.com/github/preste-ai/Macrophage-phenotype-classification/blob/main/Notebooks/Dim_reduction_analysis/tSNE_UMAP_nucleus_CD86_smooth.ipynb
https://colab.research.google.com/github/preste-ai/Macrophage-phenotype-classification/blob/main/Notebooks/Dim_reduction_analysis/tSNE_UMAP_surfaces_comparison.ipynb

Data preprocessing 
https://colab.research.google.com/github/preste-ai/Macrophage-phenotype-classification/blob/main/Notebooks/Data_preprocessing/Data_preprocessing_20x_CD206.ipynb
https://colab.research.google.com/github/preste-ai/Macrophage-phenotype-classification/blob/main/Notebooks/Data_preprocessing/Data_preprocessing_20x_CD68.ipynb
https://colab.research.google.com/github/preste-ai/Macrophage-phenotype-classification/blob/main/Notebooks/Data_preprocessing/Data_preprocessing_20x_CD86.ipynb

Classification
https://colab.research.google.com/github/preste-ai/Macrophage-phenotype-classification/blob/main/Notebooks/Classification_models/20x_CD206+CD86_P4G4_model_TCPS_data_nucleus_deformation.ipynb
https://colab.research.google.com/github/preste-ai/Macrophage-phenotype-classification/blob/main/Notebooks/Classification_models/20x_CD206+CD86_P4G4_nucleus_deformation.ipynb
https://colab.research.google.com/github/preste-ai/Macrophage-phenotype-classification/blob/main/Notebooks/Classification_models/20x_CD206+CD86_TCPS_model_P4G4_data_nucleus_deformation.ipynb
https://colab.research.google.com/github/preste-ai/Macrophage-phenotype-classification/blob/main/Notebooks/Classification_models/20x_CD206+CD86_TCPS_nucleus_deformation.ipynb
https://colab.research.google.com/github/preste-ai/Macrophage-phenotype-classification/blob/main/Notebooks/Classification_models/20x_CD206_all_surfaces_nucleus_deformation.ipynb
https://colab.research.google.com/github/preste-ai/Macrophage-phenotype-classification/blob/main/Notebooks/Classification_models/20x_CD206_nucleus_deformation.ipynb
https://colab.research.google.com/github/preste-ai/Macrophage-phenotype-classification/blob/main/Notebooks/Classification_models/20x_CD86_CD206_all_surfaces_nucleus_deformation.ipynb
https://colab.research.google.com/github/preste-ai/Macrophage-phenotype-classification/blob/main/Notebooks/Classification_models/20x_CD86_CD206_combined_nucleus_deformation.ipynb
https://colab.research.google.com/github/preste-ai/Macrophage-phenotype-classification/blob/main/Notebooks/Classification_models/20x_CD86_all_surfaces_nucleus_deformation.ipynb
https://colab.research.google.com/github/preste-ai/Macrophage-phenotype-classification/blob/main/Notebooks/Classification_models/20x_CD86_nucleus_deformation.ipynb
https://colab.research.google.com/github/preste-ai/Macrophage-phenotype-classification/blob/main/Notebooks/Classification_models/CD206_data_on_CD86_model.ipynb

## Final model's location:
| Model name | Model's location |
| ---------- | ---------------- |
| 20x_CD206_all_surfaces_all_features | 20x_CD206_all_surfaces_nucleus_deformation (1st model) |
| 20x_CD206_all_surfaces_shape_texture | 20x_CD206_all_surfaces_nucleus_deformation (2nd model) |
| 20x_CD206_all_surfaces_texture_intensity | 20x_CD206_all_surfaces_nucleus_deformation (3rd model) |
| 20x_CD206_all_surfaces_shape | 20x_CD206_all_surfaces_nucleus_deformation (4th model) |
| 20x_CD206_all_surfaces_texture | 20x_CD206_all_surfaces_nucleus_deformation (5th model)|
| 20x_CD206_all_surfaces_intensity | 20x_CD206_all_surfaces_nucleus_deformation (6th model) |
| 20x_CD206_P4G4_all_features | 20x_CD206_nucleus_deformation (1st model) |
| 20x_CD206_Smooth_all_features | 20x_CD206_nucleus_deformation (2nd model) |
| 20x_CD206_TCPS_all_features	| 20x_CD206_nucleus_deformation (3rd model) |
| 20x_CD86_all_surfaces_all_features |	20x_CD86_all_surfaces_nucleus_deformation (1st model) |
| 20x_CD86_all_surfaces_shape_texture |	20x_CD86_all_surfaces_nucleus_deformation (2nd model) |
| 20x_CD86_all_surfaces_texture_intensity |	20x_CD86_all_surfaces_nucleus_deformation (3rd model) |
| 20x_CD86_all_surfaces_shape |	20x_CD86_all_surfaces_nucleus_deformation (4th model) |
| 20x_CD86_all_surfaces_texture |	20x_CD86_all_surfaces_nucleus_deformation (5th model) |
| 20x_CD86_all_surfaces_intensity |	20x_CD86_all_surfaces_nucleus_deformation (6th model) |
| 20x_CD86_P4G4_all_features |	20x_CD86_nucleus_deformation (1st model) |
| 20x_CD86_Smooth_all_features |	20x_CD86_nucleus_deformation (2nd model) |
| 20x_CD86_TCPS_all_features |	20x_CD86_nucleus_deformation (3rd model) |
| 20x_CD86+CD206_all_surfaces_all_features |	20x_CD86_CD206_all_surfaces_nucleus_deformation (1st model) |
| 20x_CD86+CD206_all_surfaces_shape_texture |	20x_CD86_CD206_all_surfaces_nucleus_deformation (2nd model) |
| 20x_CD86+CD206_all_surfaces_texture_intensity |	20x_CD86_CD206_all_surfaces_nucleus_deformation (3rd model) |
| 20x_CD86+CD206_P4G4_all_features |	20x_CD86_CD206_combined_nucleus_deformation (1st model) |
| 20x_CD86+CD206_Smooth_all_features |	20x_CD86_CD206_combined_nucleus_deformation (2nd model) |
| 20x_CD86+CD206_TCPS_all_features |	20x_CD86_CD206_combined_nucleus_deformation (3rd model) |
| 20x_CD86_model_CD206_data_all_surfaces_all_features |	CD206_data_on_CD86_model (1st model) |
| 20x_CD206_model_CD86_data_all_surfaces_all_features |	CD86_data_on_CD206_model (1st model) |
| 20x_CD206_model_CD86_data_Smooth_all_features |	CD86_data_on_CD206_model (3rd model) |
| 20x_CD206_model_CD86_data_P4G4_all_features |	CD86_data_on_CD206_model (2nd model) |
| 20x_CD206_model_CD86_data_TCPS_all_features |	CD86_data_on_CD206_model (4th model) |
| 20x_CD86+CD206_all_surfaces_all_features_M1-M2 |	no_NT_20x_CD86_CD206 (1st model) |
| 20x_CD86_model_CD206_data_all_surfaces_all_features_M1-M2 |	no_NT_20x_CD86_CD206 (2nd model) |
| 20x_CD206_model_CD86_data_all_surfaces_all_features_M1-M2 |	no_NT_20x_CD86_CD206 (3rd model) |
| 20x_CD206+CD86_P4G4_model_TCPS_data_all_features |	20x_CD206+CD86_P4G4_model_TCPS_data_nucleus_deformation |
| 20x_CD206+CD86_TCPS_model_P4G4_data_all_features |	20x_CD206+CD86_TCPS_model_P4G4_data_nucleus_deformation |
| 20x_CD206+CD86_P4G4_intensity |	20x_CD206+CD86_P4G4_nucleus_deformation (1st model) |
| 20x_CD206+CD86_P4G4_shape |	20x_CD206+CD86_P4G4_nucleus_deformation (2nd model) |
| 20x_CD206+CD86_P4G4_texture |	20x_CD206+CD86_P4G4_nucleus_deformation (3rd model) |
| 20x_CD206+CD86_TCPS_intensity |	20x_CD206+CD86_TCPS_nucleus_deformation (1st model) |
| 20x_CD206+CD86_TCPS_shape |	20x_CD206+CD86_TCPS_nucleus_deformation (2nd model) |
| 20x_CD206+CD86_TCPS_texture |	20x_CD206+CD86_TCPS_nucleus_deformation (3rd model) |

![Alt text](NOVA-Logo_OP_Dunkel.png)
This work is funded by the European Union under grant agreement number Horizon Europe Framework Programme N° 101058554(NOVA),  The funder played no role in study design, data collection, analysis, and interpretation of data
