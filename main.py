import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class HighwayLayer(nn.Module):
    """
    Highway Layer implementasyonu.
    
    Referans Makale:
        "Highway Networks" - Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber (2015)
        arXiv:1505.00387 / arXiv:1507.06228 (NeurIPS 2015)
        
    Matematiksel Formulasyon:
        y = H(x, W_H) * T(x, W_T) + x * C(x, W_C)
        Burada C = 1 - T (bagli tasima kapisi / coupled carry gate)
        T(x) = sigmoid(W_T * x + b_T)
        H(x) = Relu(W_H * x + b_H)
    """
    def __init__(self, size: int, gate_bias: float = -2.0, activation=torch.relu):
        super().__init__()
        self.transform = nn.Linear(size, size)
        self.gate = nn.Linear(size, size)
        self.activation = activation

        # Orijinal makalede Srivastava ve ark., kapi bias'inin (b_T) negatif bir degere
        # (orn. -1, -2, -3) initialize edilmesini onerir. Boylece egitim baslangicinda
        # kapi T(x) ~ 0 olur ve ag kimlik (carry) modunda calisarak gradyanlarin
        # derin katmanlara engelsiz akmasini saglar.
        if gate_bias is not None:
            nn.init.constant_(self.gate.bias, gate_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H = self.activation(self.transform(x))
        gate = torch.sigmoid(self.gate(x))
        carry = 1.0 - gate

        output = gate * H + carry * x
        return output


class HighwayNetwork(nn.Module):
    """
    Birden fazla HighwayLayer katmanini sirali olarak baglayan derin ag mimarisi.
    """
    def __init__(self, size: int, num_layers: int, gate_bias: float = -2.0, activation=torch.relu):
        super().__init__()
        self.layers = nn.ModuleList([
            HighwayLayer(size, gate_bias=gate_bias, activation=activation)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class PlainDeepNetwork(nn.Module):
    """
    Karsilastirma icin Highway kapilari olmayan standart derin feedforward MLP agi.
    """
    def __init__(self, size: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(size, size) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


def run_vanishing_gradient_demo(num_layers: int = 20, size: int = 64, epochs: int = 150):
    """
    Derin bir Highway Network ile standart bir Plain MLP agini sentetik bir gorevde
    karsilastirarak gradyan akisi ve ogrenme basarisini test eder.
    """
    print("\n" + "=" * 65)
    print(f"DENEY: {num_layers} Katmanli Highway Network vs. Standart Duz MLP")
    print("=" * 65)

    torch.manual_seed(42)
    x = torch.randn(256, size)
    # Hedef: Girdi uzerinde dogrusal olmayan bir donusum ogrenmek
    y = torch.sin(x) + 0.1 * torch.randn_like(x)

    highway_model = HighwayNetwork(size=size, num_layers=num_layers, gate_bias=-2.0)
    plain_model = PlainDeepNetwork(size=size, num_layers=num_layers)

    criterion = nn.MSELoss()
    highway_opt = torch.optim.Adam(highway_model.parameters(), lr=1e-3)
    plain_opt = torch.optim.Adam(plain_model.parameters(), lr=1e-3)

    highway_losses = []
    plain_losses = []

    for epoch in range(epochs):
        # Highway Network Egitimi
        highway_opt.zero_grad()
        out_hw = highway_model(x)
        loss_hw = criterion(out_hw, y)
        loss_hw.backward()
        highway_opt.step()
        highway_losses.append(loss_hw.item())

        # Plain Network Egitimi
        plain_opt.zero_grad()
        out_plain = plain_model(x)
        loss_plain = criterion(out_plain, y)
        loss_plain.backward()
        plain_opt.step()
        plain_losses.append(loss_plain.item())

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:>3}/{epochs}] | Highway Loss: {loss_hw.item():.5f} | Plain Loss: {loss_plain.item():.5f}")

    print("\nSonuc:")
    print(f" - Highway Network Son Kayip  : {highway_losses[-1]:.6f}")
    print(f" - Standart Plain Ag Son Kayip: {plain_losses[-1]:.6f}")

    # Grafik olusturma ve kaydetme
    try:
        plt.figure(figsize=(9, 5))
        plt.plot(highway_losses, label="Highway Network (20 Katman)", color="#2563eb", linewidth=2)
        plt.plot(plain_losses, label="Plain MLP (20 Katman - Gradyan Sonumu)", color="#dc2626", linestyle="--", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Kaybi (Loss)")
        plt.title("Egitim Karsilastirmasi: Highway Network vs. Standart Duz Ag (20 Katman)")
        plt.yscale("log")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig("highway_vs_plain.png", dpi=150)
        plt.close()
        print("[+] Karsilastirma grafigi kaydedildi: highway_vs_plain.png")
    except Exception as e:
        print(f"Grafik kaydedilemedi: {e}")


if __name__ == "__main__":
    print("[*] Highway Network Temel Test:")
    model = HighwayNetwork(size=128, num_layers=3)
    x = torch.randn(1, 128)
    output = model(x)
    print("Girdi boyutu :", x.shape)
    print("Cikti boyutu :", output.shape)

    # Gradyan akisi ve egitim karsilastirma demosu
    run_vanishing_gradient_demo(num_layers=20, size=64, epochs=150)
