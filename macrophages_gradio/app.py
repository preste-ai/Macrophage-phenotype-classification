import gradio as gr
import pandas as pd
import re
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import inspect

# Model configuration dictionary
model_paths = {
    "20x_CD206_all_surfaces_all_features": ["models/20x_CD206_all_surfaces_all_features.joblib", "all"],
    "20x_CD206_all_surfaces_intensity": ["models/20x_CD206_all_surfaces_intensity.joblib", "intensity"],
    "20x_CD86+CD206_all_surfaces_all_features": ["models/20x_CD86+CD206_all_surfaces_all_features.joblib", "all"],
    "20x_CD86+CD206_TCPS_all_features": ["models/20x_CD86+CD206_TCPS_all_features.joblib", "all"],
    "20x_CD86+CD206_Smooth_all_features": ["models/20x_CD86+CD206_Smooth_all_features.joblib", "all"],
    "20x_CD86+CD206_P4G4_all_features": ["models/20x_CD86+CD206_P4G4_all_features.joblib", "all"],
    "20x_CD206+CD86_TCPS_intensity": ["models/20x_CD206+CD86_TCPS_intensity.joblib", "intensity"],
    "20x_CD206+CD86_P4G4_intensity": ["models/20x_CD206+CD86_P4G4_intensity.joblib", "intensity"],
    "20x_CD86+CD206_all_surfaces_all_features_M1-M2": ["models/20x_CD86+CD206_all_surfaces_all_features_M1-M2.joblib",
                                                       "all"],
}


class ModelFeatureExtractor:
    @staticmethod
    def extract_features_from_model(model_path):
        """
        Attempt to extract feature names from different types of scikit-learn models
        """
        try:
            model = joblib.load(model_path)

            # Method 1: Check for feature_names_in_ attribute (newer scikit-learn versions)
            if hasattr(model, 'feature_names_in_'):
                return list(model.feature_names_in_)

            # Method 2: Check if model has a named_steps attribute (pipeline)
            if hasattr(model, 'named_steps'):
                # Look for preprocessor or scaler that might have feature names
                for step_name, step in model.named_steps.items():
                    if hasattr(step, 'feature_names_in_'):
                        return list(step.feature_names_in_)

                    # Special handling for ColumnTransformer
                    if step_name == 'preprocessor' and hasattr(step, 'transformers_'):
                        feature_names = []
                        for _, transformer, columns in step.transformers_:
                            if hasattr(transformer, 'feature_names_in_'):
                                feature_names.extend(transformer.feature_names_in_)
                        if feature_names:
                            return feature_names

            # Method 3: Inspect the model's constructor arguments
            try:
                signature = inspect.signature(model.__class__.__init__)
                if 'feature_names' in signature.parameters:
                    return model.feature_names
            except Exception:
                pass

        except Exception as e:
            return [f"Error extracting features: {str(e)}"]

        return ["Unable to extract features"]

    @staticmethod
    def get_model_features(model_name):
        """
        Get features for a specific model
        """
        model_path = model_paths.get(model_name, [None, None])[0]
        if model_path:
            return ModelFeatureExtractor.extract_features_from_model(model_path)
        return ["Model not found"]


class ModelPredictor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.current_model_features = []

    def process_data(self, file, feature_type):
        if not file:
            raise ValueError("No file uploaded")

        data = pd.read_csv(file.name)

        if feature_type == "intensity":
            # Store features for reporting
            self.current_model_features = [col for col in data.columns
                                           if 'Intensity' in col or col == 'CellType']
            data = data[[col for col in self.current_model_features if col != 'CellType']]
        else:
            # Store all features except CellType
            self.current_model_features = [col for col in data.columns if col != 'CellType']
            data = data[[col for col in self.current_model_features]]

        return data

    def prepare_features(self, data, has_labels=False):
        if has_labels and 'CellType' not in data.columns:
            raise ValueError("CellType column not found in data")

        if has_labels:
            X = data.drop('CellType', axis=1)
            y = self.label_encoder.fit_transform(data['CellType'])
            original_labels = data['CellType'].values
            return X, y, original_labels
        else:
            return data, None, None

    def predict(self, model_name, file):
        try:
            # Get model configuration
            model_path, feature_type = model_paths.get(model_name, [None, None])
            if not model_path:
                return f"Error: Invalid model name '{model_name}'", None, None, []

            # Load model
            try:
                model = joblib.load(model_path)
            except Exception as e:
                return f"Error loading model: {str(e)}", None, None, []

            # Extract features from the model
            model_features = ModelFeatureExtractor.get_model_features(model_name)

            # Process and prepare data
            data = pd.read_csv(file.name)

            # Check if CellType is present
            has_labels = 'CellType' in data.columns

            # Process data based on feature type and label presence
            processed_data = self.process_data(file, feature_type)

            # Prepare features (with or without labels)
            X_scaled, y, original_labels = self.prepare_features(
                data if has_labels else processed_data,
                has_labels
            )

            # Make predictions
            y_pred = model.predict(X_scaled)

            # Determine results presentation based on label presence
            if has_labels:
                # Labeled prediction scenario
                predicted_labels = self.label_encoder.inverse_transform(y_pred)
                results_df = pd.DataFrame({
                    'Original Label': original_labels,
                    'Predicted Label': predicted_labels,
                    'Correct': original_labels == predicted_labels
                })

                # Add confidence scores if model supports predict_proba
                try:
                    probabilities = model.predict_proba(X_scaled)
                    confidence_scores = np.max(probabilities, axis=1)
                    results_df['Confidence'] = confidence_scores.round(3)
                except:
                    results_df['Confidence'] = 'N/A'

                # Calculate summary statistics
                total_samples = len(results_df)
                correct_predictions = results_df['Correct'].sum()
                accuracy = correct_predictions / total_samples

                # Add per-class accuracy
                class_accuracy = results_df.groupby('Original Label')['Correct'].agg(['count', 'sum', 'mean']).round(3)
                class_accuracy.columns = ['Total', 'Correct', 'Accuracy']

                # Create detailed summary
                summary_text = (
                    f"Prediction Summary:\n"
                    f"Total samples: {total_samples}\n"
                    f"Overall accuracy: {accuracy:.2%}\n\n"
                    f"Per-class Performance:\n"
                    f"Total\tCorrect\tAccuracy\n"
                    f"{class_accuracy.to_string()}"
                )

            else:
                # Unlabeled prediction scenario
                predicted_labels = model.predict(X_scaled)

                # Try to get confidence if possible
                try:
                    probabilities = model.predict_proba(X_scaled)
                    confidence_scores = np.max(probabilities, axis=1)
                    results_df = pd.DataFrame({
                        'Predicted Label': predicted_labels,
                        'Confidence': confidence_scores.round(3)
                    })
                except:
                    results_df = pd.DataFrame({
                        'Predicted Label': predicted_labels
                    })

                summary_text = (
                    f"Prediction Results:\n"
                    f"Total samples: {len(results_df)}\n"
                    f"Prediction completed without ground truth labels"
                )

            # Save results to CSV
            temp_csv_path = f"results_{model_name}.csv"
            results_df.to_csv(temp_csv_path, index=False)

            # Use model-derived features if available, otherwise fall back to current_model_features
            display_features = model_features if model_features else self.current_model_features

            return summary_text, results_df, temp_csv_path, display_features

        except Exception as e:
            return f"Error during prediction: {str(e)}", None, None, []


def create_interface():
    predictor = ModelPredictor()

    def update_model_features(model_name):
        """Callback to update features when model is selected"""
        features = ModelFeatureExtractor.get_model_features(model_name)
        return "\n".join(features)

    with gr.Blocks(css="footer {visibility: hidden}") as interface:
        with gr.Row():
            gr.HTML("""
                <div style="display: flex; justify-content: space-between; width: 100%; padding: 10px;">
                    <img src="https://docs.preste.ai/Preste-color.png" style="height: 50px; object-fit: contain;">
                    <img src="https://eu-nova.eu/skin/nova/assets/img/logo.png" style="height: 50px; object-fit: contain;">
                </div>
            """)

        gr.Markdown("# Cell Type Prediction")

        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=list(model_paths.keys()),
                label="Select Model",
                value=list(model_paths.keys())[0]
            )

        features_list = gr.Textbox(
            label="Features Used by Model",
            interactive=False,
            lines=5
        )

        model_dropdown.change(
            fn=update_model_features,
            inputs=[model_dropdown],
            outputs=[features_list]
        )

        with gr.Row():
            file_input = gr.File(
                label="Upload Dataset (CSV)",
                file_types=[".csv"]
            )

        with gr.Row():
            predict_btn = gr.Button("Make Prediction", variant="primary")

        with gr.Row():
            output_text = gr.Textbox(
                label="Summary",
                interactive=False,
                lines=10
            )

        with gr.Row():
            results_table = gr.Dataframe(
                headers=['Original Label', 'Predicted Label', 'Correct', 'Confidence'],
                label="Prediction Results",
                interactive=False,
                wrap=True
            )

        with gr.Row():
            download_btn = gr.File(
                label="Download Results",
                file_types=[".csv"]
            )

        predict_btn.click(
            fn=predictor.predict,
            inputs=[model_dropdown, file_input],
            outputs=[output_text, results_table, download_btn, features_list]
        )

    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860)
