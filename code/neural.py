from model_params import n
import numpy as np

results_doc = []
for _ in range(0, 100):
    from model import training_pairs
    results_doc = results_doc + training_pairs

results_docs = [[[ 0.13419576, 21.1887173],
 [-3.99852032, -3.97426419],
 [-4.39994205, -4.39497996],
 [-4.39996582, -4.39699213],
 [-4.39996593, -4.39699225],
 [-4.39996593, -4.39699225],
 [-4.39996593, -4.39699225],
 [-4.39996593, -4.39699225],
 [-4.39996593, -4.39699225],
 [-4.39996593, -4.39699225],
 [-4.39996593, -4.39699225]],
[[ 7.8538794,  23.41230307],
 [-3.71808042, -3.98361007],
 [-4.38991492, -4.39836543],
 [-4.39403031, -4.39922208],
 [-4.39403881, -4.39922733],
 [-4.39403886, -4.39922734],
 [-4.39403887, -4.39922734],
 [-4.39403887, -4.39922734],
 [-4.39403887, -4.39922734],
 [-4.39403887, -4.39922734],
 [-4.39403887, -4.39922734]],

[[15.34852195, 11.81208337],
 [-3.83351269, -3.87603145],
 [-4.38886952, -4.39289345],
 [-4.39398792, -4.39627683],
 [-4.39402143, -4.39630801],
 [-4.39402174, -4.39630822],
 [-4.39402174, -4.39630822],
 [-4.39402174, -4.39630822]],

[[16.83752832, 21.48290304],
 [-3.85545896, -3.9054783 ],
 [-4.39402621, -4.39504596],
 [-4.39669822, -4.3973771 ],
 [-4.39671094, -4.39738866],
 [-4.39671101, -4.39738872],
 [-4.39671101, -4.39738872]],
[[17.98231987, -2.45992697],
 [-3.98824843, -4.2434027 ],
 [-4.39649962, -4.39310611],
 [-4.39719222, -4.39588786],
 [-4.39720509, -4.39589258],
 [-4.39720511, -4.39589267],
 [-4.39720512, -4.39589267]],
[[22.98208896,  0.02931804],
 [-3.97618002, -3.90364441],
 [-4.39480741, -4.3964745 ],
 [-4.39714157, -4.39791604],
 [-4.39714839, -4.39792407],
 [-4.39714843, -4.3979241 ],
 [-4.39714843, -4.3979241 ]]
               ]
sample_pairs_cut = []

for results in results_docs:
    sample_pairs = [(np.array(results[i]), np.array(results[i+1])) for i in range(len(results) - 1)]

    flag = True
    for i in range(len(sample_pairs)):
        if all(np.isclose(sample_pairs[i][0], sample_pairs[i][1], rtol=1e-08, atol=1e-08)):
            if flag:
                sample_pairs_cut = sample_pairs_cut + sample_pairs[:i]
                flag = False

### ----- Neural Network stuff ------ ###


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        x = F.relu(self.fc2(x))
        return x


# Custom loss function
def custom_loss(V_x, f_x_prime, epsilon):
    return torch.sum(F.relu(f_x_prime - V_x + epsilon))


# Example usage
if __name__ == '__main__':
    input_size = n
    hidden_size = 10
    output_size = 1
    #num_samples = len(sample_pairs_cut)
    epsilon = 0.000001
    learning_rate = 0.0000001
    num_epochs = 1000

    # Instantiate the neural network
    model = SimpleNN(input_size, hidden_size, output_size)

    X = torch.tensor(np.array([i[0][0] for i in results_doc]), dtype=torch.float32)
    X_prime = torch.tensor(np.array([i[1] for i in results_doc]), dtype=torch.float32).permute(1,0,2)


    #print(X, X_prime)
    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        V_x = model(X)
        V_x_prime = torch.stack([model(i) for i in X_prime]).permute(1, 0, 2)
        E_V_x_prime = torch.mean(V_x_prime, dim=1)

        # Compute loss
        loss = custom_loss(V_x, E_V_x_prime, epsilon)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        f_x = model(X)
        f_x_prime = model(X_prime)
        final_loss = custom_loss(f_x, f_x_prime, epsilon)
        print(f'Final Loss: {final_loss.item():.4f}')

