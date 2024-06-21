"""
Modelling the ODE dT/dt=-coef*∆T with PINN

Exact solution: T(t) = (TEMP_0 - TEMP_AMBIENT) * exp(-coef*t)

This time, using a ResNet style architecture with residual connections.
There is a parallel to Euler integration.

TODO:
 - Make coef, temp_0, temp_ambient inputs to the network
"""

import numpy as np
import plotly.graph_objects as go
from plotly import subplots
import torch
import tqdm

PLOTLY_TEMPLATE = "plotly_dark"

TRAINING_T_RANGE = (0, 2)
TESTING_T_RANGE = (0, 5)

COEF = 2
TEMP_0 = 50  # deg C
TEMP_AMBIENT = 20  # deg C


class ResidualBlock(torch.nn.Module):
    """Define a residual block"""

    def __init__(self, hidden_dim: int):
        super(ResidualBlock, self).__init__()
        self.linear = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.activation1 = torch.nn.Tanh()

    def forward(self, x):
        """Forward pass"""
        residual = x
        x = self.linear(x)
        # This makes a massive difference to generalisation, compare to the previous PINN
        x += residual
        x = self.activation1(x)
        x = torch.nn.functional.layer_norm(x, x.shape)
        return x


class DemoPINN(torch.nn.Sequential):
    """Define the simple 1 layer network"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_residual_blocks: int,
        hidden_dim: int = 32,
    ):
        residuals = self._make_residual_block(hidden_dim, num_residual_blocks)
        super(DemoPINN, self).__init__(
            torch.nn.Linear(in_features=input_dim, out_features=hidden_dim),
            *residuals,
            torch.nn.Linear(in_features=hidden_dim, out_features=output_dim),
        )

    def _make_residual_block(
        self, hidden_dim: int, num_residual_blocks: int
    ) -> list[torch.nn.Module]:
        blocks = []
        for _ in range(num_residual_blocks):
            blocks.append(ResidualBlock(hidden_dim))
        return blocks


def pinn_loss(
    t: np.ndarray,
    coef: float,
    temp_0: float,
    temp_ambient: float,
    model: torch.nn.Module,
):
    """Loss function for the PINN"""

    # Convert t to tensor
    t = t.reshape(-1, 1)
    t = torch.tensor(t, dtype=torch.float32, requires_grad=True)

    # Initial conditions
    temp_0: torch.Tensor = torch.ones(1) * temp_0
    temp_ambient: torch.Tensor = torch.ones(1) * temp_ambient
    t_0: torch.Tensor = torch.zeros(1)
    # one: torch.Tensor = torch.ones(1)

    # Make the prediction
    temp_pred: torch.Tensor = model(t)

    # Use autograd to compute the derivative of y_pred wrt t
    temp_pred_dt: torch.Tensor = torch.autograd.grad(
        outputs=temp_pred,
        inputs=t,
        grad_outputs=torch.ones_like(temp_pred),
        create_graph=True,
    )[0]  # take first an only element of the tuple (gradients wrt t)

    # Loss is the square of the difference between
    # the predicted derivative and the actual derivative
    ode_loss = temp_pred_dt + coef * (temp_pred - temp_ambient)
    initial_condition_loss = model(t_0) - temp_0

    loss = torch.mean(torch.pow(ode_loss, 2) + torch.pow(initial_condition_loss, 2))
    return loss


def exact_solution(t, coef, temp_0, temp_ambient):
    """Returns the exact solution to the ODE dT/dt = -∆T"""
    return (temp_0 - temp_ambient) * np.exp(-coef * t) + temp_ambient


def noisy_exact_solution(t, coef, temp_0, temp_ambient, noise_std=0.5):
    """Returns the exact solution to the ODE dT/dt = -∆T with added noise"""
    return exact_solution(t, coef, temp_0, temp_ambient) + np.random.normal(
        0, noise_std, size=t.shape
    )


def main():
    """Train the PINN and evaluate it against the exact solution"""

    # Define the model
    demo_pinn = DemoPINN(
        input_dim=1,
        output_dim=1,
        num_residual_blocks=3,
        hidden_dim=32,
    )

    # Define the optimizer and basic (non-physics informed) loss function
    optimizer = torch.optim.Adam(demo_pinn.parameters(), lr=0.01)
    basic_loss = torch.nn.MSELoss()

    # Training loop
    train_losses = []
    for _ in tqdm.trange(1_000):
        t_train = np.random.rand(100) * TRAINING_T_RANGE[1]
        y_train = torch.tensor(
            noisy_exact_solution(
                t_train, coef=COEF, temp_0=TEMP_0, temp_ambient=TEMP_AMBIENT
            ),
            dtype=torch.float32,
        ).reshape(-1, 1)

        demo_pinn.train()
        optimizer.zero_grad()
        y_pred = demo_pinn(torch.tensor(t_train, dtype=torch.float32).reshape(-1, 1))
        loss_basic = basic_loss(y_pred, y_train)
        loss_ode = pinn_loss(
            t=t_train,
            coef=COEF,
            temp_0=TEMP_0,
            temp_ambient=TEMP_AMBIENT,
            model=demo_pinn,
        )

        loss_val = loss_basic + loss_ode  # Experiment with removing the ODE loss

        train_losses.append(loss_val.item())
        loss_val.backward()
        optimizer.step()

    # Evaluate the model
    t_test = np.linspace(TESTING_T_RANGE[0], TESTING_T_RANGE[1], 100)

    demo_pinn.eval()
    with torch.no_grad():
        y_pred: torch.Tensor = demo_pinn(
            torch.tensor(t_test, dtype=torch.float32).reshape(-1, 1)
        )
        y_pred = y_pred.squeeze().numpy()
    y_exact = exact_solution(
        t_test, coef=COEF, temp_0=TEMP_0, temp_ambient=TEMP_AMBIENT
    )

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
        go.Scatter(x=t_test, y=y_pred, mode="lines", name="Predicted"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=t_test, y=y_exact, mode="lines", name="Exact"),
        row=1,
        col=2,
    )
    fig.add_vrect(
        x0=TRAINING_T_RANGE[0], x1=TRAINING_T_RANGE[1], fillcolor="green", opacity=0.2
    )

    fig.update_layout(title="PINN vs Exact solution", template=PLOTLY_TEMPLATE)
    fig.show()

    print("Model", demo_pinn)

    print(
        "Test Loss:",
        f"{basic_loss(torch.tensor(y_pred), torch.tensor(y_exact)).item():,.4f}",
    )


if __name__ == "__main__":
    main()
