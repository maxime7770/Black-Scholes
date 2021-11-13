import pandas as pd
import matplotlib.pyplot as plt


validations = pd.read_csv('tableResultsValidaton.csv')


plt.plot(validations['Full'], label='model_full')
plt.plot(validations['Sparse'], label='mode_sparse')
plt.plot(validations['Extremes'], label='model_extremes')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('validation error')
plt.title('Validation errors for the different models')
plt.show()

plt.plot(validations['Full'])
plt.title('Validation erros for the full model')
plt.xlabel('epochs')
plt.ylabel('validation error')
plt.show()
