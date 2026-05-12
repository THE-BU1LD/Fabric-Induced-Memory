from __future__ import annotations

import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def run(model, benchmark, device="cpu"):
    device = torch.device(device)
    model.eval()

    x, y = benchmark.generate_batch(batch_size=64, device=device)

    x = x.unsqueeze(2).unsqueeze(3).to(device)
    y = y.unsqueeze(2).unsqueeze(3).to(device)

    preds = []
    latents_full = []
    latents_flat = []
    saliences = []

    with torch.no_grad():
        for t in range(x.shape[1]):
            pred, sal = model(x[:, t])

            preds.append(pred.squeeze(-1).squeeze(-1).cpu())
            saliences.append(sal.cpu())

            z = model.encoder(x[:, t])
            z = model.dynamics(z)

            latents_full.append(z.cpu())
            latents_flat.append(z.mean(dim=(2, 3)).cpu())

    preds = torch.stack(preds, dim=1)
    latents_full = torch.stack(latents_full, dim=1)
    latents_flat = torch.cat(latents_flat, dim=0)
    saliences = torch.stack(saliences, dim=1)

    true = y.squeeze(-1).squeeze(-1).cpu()
    pred = preds.cpu()

    mse = torch.mean((pred - true) ** 2).item()

    plt.figure()
    plt.plot(pred[0, :, 0].numpy())
    plt.plot(true[0, :, 0].numpy())
    plt.title("Trajectory")
    plt.savefig("trajectory.png")
    plt.close()

    pca = PCA(n_components=3)
    z_3d = pca.fit_transform(latents_flat.numpy())

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(z_3d[:, 0], z_3d[:, 1], z_3d[:, 2], s=2)
    plt.title("Latent Manifold")
    plt.savefig("latent_manifold.png")
    plt.close()

    plt.figure()
    sal_map = saliences[0, -1, 0]
    plt.imshow(sal_map.numpy())
    plt.colorbar()
    plt.title("Spatial Salience")
    plt.savefig("salience_spatial.png")
    plt.close()

    plt.figure()
    latent_map = latents_full[0, -1, 0]
    plt.imshow(latent_map.numpy())
    plt.colorbar()
    plt.title("Latent Heatmap")
    plt.savefig("latent_heatmap.png")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    Z = latent_map.numpy()
    H, W = Z.shape

    X, Y = torch.meshgrid(
        torch.arange(H),
        torch.arange(W),
        indexing="ij"
    )

    ax.plot_surface(X.numpy(), Y.numpy(), Z)
    plt.title("3D Surface")
    plt.savefig("latent_surface.png")
    plt.close()

    diff = torch.abs(pred - true).mean(dim=-1)

    plt.figure()
    plt.plot(diff[0].numpy())
    plt.title("Chaos Divergence")
    plt.savefig("chaos.png")
    plt.close()

    latent_energy = latents_flat.norm(dim=1)

    plt.figure()
    plt.plot(latent_energy.numpy())
    plt.title("Latent Energy")
    plt.savefig("energy.png")
    plt.close()

    sal_scalar = saliences.mean(dim=(2, 3, 4))

    plt.figure()
    plt.imshow(sal_scalar.numpy(), aspect="auto")
    plt.colorbar()
    plt.title("Salience Over Time")
    plt.savefig("salience_time.png")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    T = latents_full.shape[1]
    H = latents_full.shape[3]
    W = latents_full.shape[4]

    for t in range(0, T, max(1, T // 20)):
        surface = latents_full[0, t, 0].numpy()

        X, Y = torch.meshgrid(
            torch.arange(H),
            torch.arange(W),
            indexing="ij"
        )

        ax.plot_surface(
            X.numpy(),
            Y.numpy(),
            surface,
            alpha=0.3
        )

    plt.title("3D Time Evolution")
    plt.savefig("time_evolution_3d.png")
    plt.close()

    return {
        "mse": mse
    }