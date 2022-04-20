class HTO():

    def objective(self, Z):
        """
        Returns the scalar value of the objective given x.
        """
        return 0

    def gradient(self, Z):
        """
        Returns the gradient of the objective with respect to x. Should be a 1-dim numpy array
        """
        return 0

    def constraints(self, Z):
        """
        Returns the constraints. Should be a 1-dim numpy array
        """
        return 0

    def jacobian(self, Z):
        """
        Returns the Jacobian of the constraints with respect to x. Should be a 1-dim numpy array
        """
        return 0

    def hessian(self, Z, lagrange, obj_factor):
        """
        Returns the non-zero values of the Hessian. Should be a 2-dim numpy array (?)
        """
        return 0
