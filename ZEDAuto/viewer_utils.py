try:
    import open3d as o3d

    HAS_OPEN3D = True
except Exception:
    HAS_OPEN3D = False


class MeshViewer:
    def __init__(self, title="ZED Spatial Mesh"):
        self.vis = None
        self.mesh = None
        self.title = title

    def open(self):
        if not HAS_OPEN3D:
            print("Open3D not available. Install open3d to enable mesh viewer.")
            return False
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=self.title, width=960, height=720)
        return True

    def update_from_path(self, mesh_path):
        if self.vis is None:
            return
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if mesh is None or len(mesh.vertices) == 0:
            return
        mesh.compute_vertex_normals()
        if self.mesh is None:
            self.mesh = mesh
            self.vis.add_geometry(self.mesh)
        else:
            self.mesh.vertices = mesh.vertices
            self.mesh.triangles = mesh.triangles
            self.mesh.vertex_normals = mesh.vertex_normals
            self.vis.update_geometry(self.mesh)

    def poll(self):
        if self.vis is None:
            return
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        if self.vis is not None:
            self.vis.destroy_window()
            self.vis = None
