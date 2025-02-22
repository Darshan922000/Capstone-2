Time-Series
```mermaid
graph TD;
    A[Load Data ] --> B[Exploratory Data Analysis]
    B --> C[Preprocess Data]
    C --> D[Feature Engineering]
    D --> E[Split Data into Train & Test]
    E --> F[Model Training]
    F --> G[Hyperparameter Tuning & Feature Engineering for Improving Accuracy]
    G --> H[Evaluate Model Performance]
    H --> I{Is Model Performing Well?}
    I -- Yes --> J[Make Forecasts]
    I -- No --> G
    J --> K[Select Best Model for Forecasting]
    K --> L[Deployment]
