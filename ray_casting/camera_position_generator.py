import numpy as np

"""
This code generate a set of 3D camera poses on an unit sphere, represented by a matrix of size N x 3,
with N being number of cameras, and each row is a 3D position of a camera.

similarity describes how nearby the generated camera positions are.
high means very nearby, low means very not nearby, medium lies between them
"""


def cam_generator(similarity, n_cam_pos=3):

    def _cam_generator(view_point_range, number_of_cam=2):

        ## alpha and phi are the yaw and pitch respectively in the Euler angle.
        alpha = np.random.rand(number_of_cam) * (2 * view_point_range)                     # now in range [0, 2*view_point_range]
        phi   = np.random.rand(number_of_cam) * view_point_range  - view_point_range / 2.0 # now in range [-view_point_range/2, view_point_range/2]
        alpha += np.random.rand(1) * 2 * (np.pi - view_point_range)   # create a random deviation, now in range [deviation, 2*view_point_range + deviation]
        alpha = np.clip(alpha, 0, 2*np.pi)   # restrict alpha in range [0, 2*pi]
        if np.random.rand(1) > 0.5:
            phi   += np.random.rand(1) * (np.pi - view_point_range) / 2
        else:
            phi   -= np.random.rand(1) * (np.pi - view_point_range) / 2
        phi   = np.clip(phi, -np.pi/2, np.pi/2)
        z = np.sin(phi)
        x,y = np.cos(phi) * np.sin(alpha), np.cos(phi) * np.cos(alpha)

        points = np.column_stack((x[...,None], y[...,None], z[...,None]))

        return points / np.linalg.norm(points, axis=1, keepdims=True)

    def generate_and_concatenate(num_arrays, view_point_range):
        arrays = [_cam_generator(view_point_range=view_point_range) for _ in range(num_arrays)]
        return np.stack(arrays, axis=1)

    if similarity == "high":
        result = generate_and_concatenate(n_cam_pos, view_point_range=np.pi/8)
        return result
    elif similarity == "medium":
        return generate_and_concatenate(n_cam_pos, view_point_range=np.pi / 4)
    elif similarity == "low":
        return np.dstack([_cam_generator(view_point_range=np.pi) for _ in range(n_cam_pos)])
    else:
        raise ValueError("Undefined Type!")
