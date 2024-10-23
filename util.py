import sys

import numpy as np

# http://www.scipy.org/
try:
    from numpy import dot
    from numpy.linalg import norm
except:
    print("Error: Requires numpy from http://www.scipy.org/. Have you installed scipy?")
    sys.exit()


def removeDuplicates(list):
    """remove duplicates from a list"""
    return set((item for item in list))


def cosine(vector1, vector2):
    """Calculate cosine similarity between two vectors.
    Returns 0.0 if either vector has zero magnitude."""
    try:
        # Convert to numpy arrays if they aren't already
        v1 = np.array(vector1)
        v2 = np.array(vector2)

        n1 = norm(v1)
        n2 = norm(v2)

        # Check for zero vectors to avoid division by zero
        if n1 == 0 or n2 == 0:
            return 0.0

        return float(dot(v1, v2) / (n1 * n2))
    except Exception as e:
        print(f"Error in cosine similarity calculation: {e}")
        return 0.0
