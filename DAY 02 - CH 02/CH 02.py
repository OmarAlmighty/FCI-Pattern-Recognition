#%%
import numpy as np
import matplotlib.pyplot as plt

from prml.rv import (Bernoulli,
                     Beta)

np.random.seed(1234)
#%%
data = np.array([0, 1,1,1,1])
model = Bernoulli()
model.fit(data)
print("The estimated mu is: ",
      model.mu)
#%%
for i in range(3):
    print("Experiment", i,":",
          model.draw(1))
    