Also available on:
https://unioulu-my.sharepoint.com/:f:/g/personal/mshafiq24_student_oulu_fi/ErK-_19ryMlDuHDJRG5SqFgBOjYIuWwlpN1vyBtq4DUoqw?e=kimEtz

# Diabetic Retinopathy Detection

A deep learning project for automated detection and grading of diabetic retinopathy from retinal images.

## Checkpoints

Model checkpoints (.pth) are available in the results folder.

## Results

CSV results for each part are available in the results folder.

## Visualizations

Kappa and accuracy plots are available in the plots folder. Gradcam visualizations of each fine-tuned model are available in gradcam_results folder.

## Project Structure

The project is divided into different parts, each focusing on specific aspects of the solution:

- `part_a.py` - Initial model fine-tuning on DeepDRiD dataset
- `part_b.py` - Two-stage transfer learning with additional datasets
- `part_c_channel.py` - Implementation with channel attention mechanism
- `part_c_spatial.py` - Implementation with spatial attention mechanism
- `part_d_simple.py` - Ensemble learning implementation
- `part_d_simple_preprocessing.py` - Image preprocessing techniques
- `part_e.py` - Visualizations and model interpretability

## Running the Code

Each part can be run independently. For example:

```python
python part_a.py  # Run the initial fine-tuning
python part_b.py  # Run two-stage transfer learning
```
