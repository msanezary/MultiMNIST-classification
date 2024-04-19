# MultiMNIST-classification

This project implements a multi-task learning model for the MultiMNIST dataset. It uses a shared CNN base to predict two digits simultaneously from overlaid MNIST images.

## Project Structure

MultiMNIST/
│
├── src/                      # Source files
│   ├── main.py               # Entry point for training and evaluation
│   ├── models.py             # Model definitions
│   ├── dataset.py            # Dataset handling
│   ├── train.py              # Training routines
│   ├── evaluate.py           # Evaluation routines
│   └── utils.py              # Utility functions
│
├── data/                     # Dataset directory
│   └── multi_mnist.pickle    # Serialized dataset file
│
├── notebooks/                # Jupyter notebooks for exploration
│   └── exploration.ipynb
│
├── outputs/                  # Outputs including logs and saved models
│
├── requirements.txt          # Python dependencies
├── README.md                 # Overview and instructions
└── .gitignore                # Specifies untracked files to ignore

## Setup

To run this project, follow these steps:

### Requirements

You'll need Python 3.8+ installed on your system. It's recommended to use a virtual environment.

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/multimnist.git
   cd multimnist
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Data

Ensure the `multi_mnist.pickle` file is placed in the `data/` directory. You can modify `src/dataset.py` if your dataset is structured differently.

## Usage

To run the training and evaluation, execute the `main.py` script:

```bash
python src/main.py
```

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is open-sourced under the MIT License. See the LICENSE file for more details.

## Contact

For questions or support, please contact [msanezary@gmail.com](mailto:msanezary@gmail.com) [boutzil.50@gmail.com](mailto:boutzil.50@gmail.com).