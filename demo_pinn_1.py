"""
Modelling the ODE dy/dt=cos2πt with PINN

Exact solution: y(t) = 1/2*sin(2πt) + 1

Following the tutorial https://i-systems.github.io/tutorial/KSME/CAE/220520/01_PINN.html
"""

import numpy as np
import plotly.graph_objects as go
from plotly import subplots
import torch
import tqdm

PLOTLY_TEMPLATE = "plotly_dark"

TRAINING_T_RANGE = (0, 2)
TESTING_T_RANGE = (0, 5)


class DemoPINN(torch.nn.Module):
    """Define the simple 1 layer network"""

    def __init__(self):
        super(DemoPINN, self).__init__()
        self.linear1 = torch.nn.Linear(in_features=1, out_features=32)
        self.activation1 = torch.nn.Tanh()
        self.output = torch.nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        """Forward pass"""
        x = self.linear1(x)
        x = self.activation1(x)
        y = self.output(x)
        return y


def loss(t: np.ndarray, model: torch.nn.Module):
    """Loss function for the PINN"""

    # Convert t to tensor
    t = t.reshape(-1, 1)
    t = torch.tensor(t, dtype=torch.float32, requires_grad=True)

    # Initial conditions
    t_0: torch.Tensor = torch.zeros(1)
    one: torch.Tensor = torch.ones(1)

    # Make the prediction
    y_pred: torch.Tensor = model(t)

    # Use autograd to compute the derivative of y_pred wrt t
    y_pred_dt: torch.Tensor = torch.autograd.grad(
        outputs=y_pred,
        inputs=t,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,
    )[0]  # take first an only element of the tuple (gradients wrt t)

    # Loss is the square of the difference between the
    # predicted derivative and the actual derivative
    ode_loss = y_pred_dt - torch.cos(2 * np.pi * t)
    initial_condition_loss = model(t_0) - one
    square_loss = torch.pow(ode_loss, 2) + torch.pow(initial_condition_loss, 2)
    total_loss = torch.mean(square_loss)

    return total_loss


def exact_solution(t):
    """Returns the exact solution to the ODE dy/dt = cos(2πt) with y(0) = 1"""
    return np.sin(2 * np.pi * t) / (2 * np.pi) + 1


def main():
    """Train the PINN and evaluate it against the exact solution"""
    demo_pinn = DemoPINN()
    optimizer = torch.optim.Adam(demo_pinn.parameters(), lr=0.01)

    # Training loop
    train_losses = []
    for _ in tqdm.trange(1_000):
        demo_pinn.train()
        optimizer.zero_grad()
        loss_val = loss(np.random.rand(30) * TRAINING_T_RANGE[1], demo_pinn)
        train_losses.append(loss_val.item())
        loss_val.backward()
        optimizer.step()

    # Evaluate the model
    t = np.linspace(TESTING_T_RANGE[0], TESTING_T_RANGE[1], 100)

    demo_pinn.eval()
    with torch.no_grad():
        y_pred: torch.Tensor = demo_pinn(
            torch.tensor(t, dtype=torch.float32).reshape(-1, 1)
        )
        y_pred = y_pred.squeeze().numpy()
    y_exact = exact_solution(t)

    fig = subplots.make_subplots(
        rows=1, cols=2, subplot_titles=("Training Loss", "PINN vs Exact Solution")
    )

    # Plot the training loss
    fig.add_trace(
        go.Scatter(
            x=list(range(len(train_losses))),
            y=train_losses,
            name="Training Loss",
            mode="lines",
        ),
        row=1,
        col=1,
    )

    # Plot the PINN predictions vs the exact solution
    fig.add_trace(
        go.Scatter(x=t, y=y_pred, mode="lines", name="Predicted"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=t, y=y_exact, mode="lines", name="Exact"),
        row=1,
        col=2,
    )
    fig.add_vrect(
        x0=TRAINING_T_RANGE[0], x1=TRAINING_T_RANGE[1], fillcolor="green", opacity=0.2
    )

    fig.update_layout(title="PINN vs Exact solution", template=PLOTLY_TEMPLATE)
    fig.show()


if __name__ == "__main__":
    main()
