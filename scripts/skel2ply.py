import trimesh
import numpy as np

class CVertex:
    def __init__(self, P=None, N=None, bIsOriginal=False, is_inner_points=False, is_skel_point=False, is_target=False):
        self.P = np.zeros(3) if P is None else P  # Position
        self.N = np.zeros(3) if N is None else N  # Normal
        self.bIsOriginal = bIsOriginal
        self.is_inner_points = is_inner_points
        self.is_skel_point = is_skel_point
        self.is_target = is_target

class Skeleton:
    def __init__(self):
        self.branches = []

    def generateBranchSampleMap(self):
        # Stub function for generating branch-sample mapping
        pass

class DataMgr:
    def __init__(self):
        self.original = []
        self.samples = []
        self.inner_points = []
        self.skel_points = []
        self.target_samples = []
        self.target_original = []
        self.target_skel_points = []
        self.skeleton = Skeleton()

    def loadSkeletonFromSkel(self, fileName):
        with open(fileName, 'r') as infile:
            lines = infile.readlines()

        sem = iter(lines)

        
        def next_line():
            try:
                return next(sem).strip()
            except StopIteration:
                return None  # Return None when there are no more lines


        while True:
            str = next_line()
            # print(str)
            if str is None:
                break  # End of file reached
            
            if "ON" in str:
                print(str)
                # num = int(next_line())
                num = int(str.split(' ')[-1])
                print(f"Loading {num} original points")
                for _ in range(num):
                    v = CVertex(bIsOriginal=True)
                    line_data = next_line().split()
                    v.P = np.array([float(x) for x in line_data[:3]])
                    v.N = np.array([float(x) for x in line_data[3:]])
                    self.original.append(v)

            elif "SN" in str:
                print(str)
                num = int(str.split(' ')[-1])
                print(f"Loading {num} sample points")
                for _ in range(num):
                    v = CVertex(bIsOriginal=False)
                    line_data = next_line().split()
                    v.P = np.array([float(x) for x in line_data[:3]])
                    v.N = np.array([float(x) for x in line_data[3:]])
                    self.samples.append(v)

            elif "SkelPN" in str:
                print(str)
                num = int(str.split(' ')[-1])
                print(f"Loading {num} skeletal points")
                for _ in range(num):
                    v = CVertex(bIsOriginal=False, is_skel_point=True)
                    line_data = next_line().split()
                    v.P = np.array([float(x) for x in line_data[:3]])
                    v.N = np.array([float(x) for x in line_data[3:]])
                    self.skel_points.append(v)

            # Handle more sections similarly...
            elif str == "END":
                break  # Assume END to terminate the file

        print(f"Total original points: {len(self.original)}")
        print(f"Total sample points: {len(self.samples)}")
        print(f"Total skeletal points: {len(self.skel_points)}")

        # Call to process the skeleton branches
        self.skeleton.generateBranchSampleMap()

        print(f"Total original points: {len(self.original)}")
        print(f"Total sample points: {len(self.samples)}")
        print(f"Total skeletal points: {len(self.skel_points)}")

        # Call to process the skeleton branches
        self.skeleton.generateBranchSampleMap()

    def write_to_ply(self, all_vertices,ply_filename):
        # Combine all vertices for simplicity, you can choose specific ones.
        # all_vertices = self.original + self.samples + self.skel_points
        if not all_vertices:
            print("No points found to write!")
            return
        points = np.array([v.P for v in all_vertices])
        normals = np.array([v.N for v in all_vertices])

        # Create a trimesh point cloud object
        cloud = trimesh.points.PointCloud(vertices=points, normals=normals)

        # Export to a PLY file
        cloud.export(ply_filename)
        print(f"Written {len(points)} points to {ply_filename}")

    def write_split(self):
        self.write_to_ply(self.original,'out_ori.ply')
        self.write_to_ply(self.samples,'out_sn.ply')
        self.write_to_ply(self.skel_points,'out_skel.ply')


# Usage example
data_mgr = DataMgr()
data_mgr.loadSkeletonFromSkel('/home/wanhu/workspace/gensdf/outputs/skel_data/2.skel')
# data_mgr.write_to_ply('output_file.ply')
data_mgr.write_split()