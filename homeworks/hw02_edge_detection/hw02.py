import numpy as np


def compute_sobel_gradients_two_loops(image):
    # Get image dimensions
    height, width = image.shape

    # Initialize output gradients
    gradient_x = np.zeros_like(image, dtype=np.float64)
    gradient_y = np.zeros_like(image, dtype=np.float64)

    # Pad the image with zeros to handle borders
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    # __________end of block__________

    # Define the Sobel kernels for X and Y gradients
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Apply Sobel filter for X and Y gradients using convolution
    for i in range(1, height + 1):
        for j in range(1, width + 1):
            im = padded_image[i - 1 : i + 2, j - 1 : j + 2]
            gradient_x[i - 1, j - 1] = np.sum(im * sobel_x)
            gradient_y[i - 1, j - 1] = np.sum(im * sobel_y)

    return gradient_x, gradient_y


def compute_gradient_magnitude(sobel_x, sobel_y):
    """
    Compute the magnitude of the gradient given the x and y gradients.

    Inputs:
        sobel_x: numpy array of the x gradient.
        sobel_y: numpy array of the y gradient.

    Returns:
        magnitude: numpy array of the same shape as the input [0] with the magnitude of the gradient.
    """
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return magnitude


def compute_gradient_direction(sobel_x, sobel_y):
    """
    Compute the direction of the gradient given the x and y gradients. Angle must be in degrees in the range (-180; 180].
    Use arctan2 function to compute the angle.

    Inputs:
        sobel_x: numpy array of the x gradient.
        sobel_y: numpy array of the y gradient.

    Returns:
        gradient_direction: numpy array of the same shape as the input [0] with the direction of the gradient.
    """
    angle_rad = np.arctan2(sobel_y, sobel_x)
    angle_deg = np.rad2deg(angle_rad)
    angle_deg[angle_deg <= -180] += 360  # to ensure angles are in (-180; 180]
    return angle_deg


cell_size = 7


def compute_hog(image, pixels_per_cell=(cell_size, cell_size), bins=9):

    # 1. Convert the image to grayscale if it's not already (assuming the image is in RGB or BGR)
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)  # Simple averaging to convert to grayscale

    # 2. Compute gradients with Sobel filter
    gradient_x, gradient_y = compute_sobel_gradients_two_loops(image=image)
    # gradient_x, gradient_y = compute_sobel_gradients_opencv(image=image)

    # 3. Compute gradient magnitude and direction
    magnitude = compute_gradient_magnitude(sobel_x=gradient_x, sobel_y=gradient_y)
    direction = compute_gradient_direction(sobel_x=gradient_x, sobel_y=gradient_y)

    # plot_image_with_cells(magnitude)
    # plot_image_with_cells(direction)

    # 4. Create histograms of gradient directions for each cell
    cell_height, cell_width = pixels_per_cell

    n_cells_x = image.shape[1] // cell_width
    n_cells_y = image.shape[0] // cell_height

    histograms = np.zeros((n_cells_y, n_cells_x, bins))

    for ncy in range(n_cells_y):
        for ncx in range(n_cells_x):
            x1 = ncx * cell_width
            x2 = (ncx + 1) * cell_width
            y1 = ncy * cell_height
            y2 = (ncy + 1) * cell_height
            mc = magnitude[y1:y2, x1:x2].flatten()
            dc = direction[y1:y2, x1:x2].flatten()
            hist = np.histogram(
                dc, bins=bins, range=(-180, 180), weights=mc, density=False
            )[0]

            if np.sum(hist) != 0:
                hist = hist / np.sum(hist)
            histograms[ncy, ncx] = hist

    return histograms


# test_image, test_hog = get_test_image_hog()
# hog = compute_hog(test_image)

# plot_image_with_cells(test_image)
# plot_hog(test_hog)
# plot_hog(hog)

# test_hog[0]
