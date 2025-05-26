import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MobiusStrip:
    def __init__(self, R=1.0, w=0.2, n=200):
        self.R = R          # Radius
        self.w = w          # Width
        self.n = n          # Resolution
        self.u, self.v = np.meshgrid(
            np.linspace(0, 2 * np.pi, n),
            np.linspace(-w / 2, w / 2, n)
        )
        self.x, self.y, self.z = self._generate_surface()

    def _generate_surface(self):
        u, v, R = self.u, self.v, self.R
        x = (R + v * np.cos(u / 2)) * np.cos(u)
        y = (R + v * np.cos(u / 2)) * np.sin(u)
        z = v * np.sin(u / 2)
        return x, y, z

    def surface_area(self):
        """
        Approximate the surface area using numerical integration.
        A ≈ ∑ ||∂x/∂u × ∂x/∂v|| du dv
        """
        du = 2 * np.pi / (self.n - 1)
        dv = self.w / (self.n - 1)

        # Compute partial derivatives
        xu, xv = np.gradient(self.x, du, dv, edge_order=2)
        yu, yv = np.gradient(self.y, du, dv, edge_order=2)
        zu, zv = np.gradient(self.z, du, dv, edge_order=2)

        # Cross product of partials
        cross_x = yu * zv - zu * yv
        cross_y = zu * xv - xu * zv
        cross_z = xu * yv - yu * xv

        dA = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)
        area = np.sum(dA) * du * dv
        return area

    def edge_length(self):
        """
        Approximate edge length by summing distances along the edge (v = ±w/2).
        """
        # Edge at v = +w/2
        edge_u = np.linspace(0, 2 * np.pi, self.n)
        edge_v = np.full(self.n, self.w / 2)
        x1 = (self.R + edge_v * np.cos(edge_u / 2)) * np.cos(edge_u)
        y1 = (self.R + edge_v * np.cos(edge_u / 2)) * np.sin(edge_u)
        z1 = edge_v * np.sin(edge_u / 2)

        # Compute distance between consecutive edge points
        dx = np.diff(x1)
        dy = np.diff(y1)
        dz = np.diff(z1)
        segment_lengths = np.sqrt(dx**2 + dy**2 + dz**2)
        return np.sum(segment_lengths) * 2  # Both edges

    def plot(self):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.x, self.y, self.z, rstride=1, cstride=1,
                        color='lightblue', edgecolor='gray', linewidth=0.1)
        ax.set_title("Möbius Strip")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    strip = MobiusStrip(R=1.0, w=0.4, n=200)
    print("Approx. Surface Area:", strip.surface_area())
    print("Approx. Edge Length:", strip.edge_length())
    strip.plot()
