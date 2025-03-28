from point import Point

class Lane:
    def __init__(self, control_points: list[Point], lane_width: float, closed_loop: bool = False):
        """Represents a lane in an environment. Based on given control points, it creates a lane with a given width.
        The lane is represented as a series of splines through the control points. The lane is created based on the closed_loop parameter.

        Args:
            control_points (list[Point]): List of Point object with x, y values to create splines through.
            lane_width (float): Width (in feet) of the lane.
            closed_loop (bool, optional): If True, lane is calculated so that the end of the lane connects with the start. Defaults to False.
        """
        self.lane_width = lane_width
        self.closed_loop = closed_loop
        self.lane_setup(control_points, lane_width, closed_loop)
        
    def lane_setup(self, control_points: list[Point], lane_width: float=None, closed_loop: bool=None):
        """Sets up the lane based on the given control points, lane width and closed loop parameter.
        
        Args:
            control_points (list[Point]): List of Point object with x, y values to create spline through.
            lane_width (float, optional): Width (in feet) of the lane. Defaults to None.
            closed_loop (bool, optional): If True, lane is calculated so that the end of the lane connects with the start. Defaults to None.
        """
        if lane_width is not None:
            self.lane_width = lane_width
        if closed_loop is not None:
            self.closed_loop = closed_loop
        
        self.center_line = self.calculate_center(control_points)
        self.left_edge, self.right_edge = self.calculate_edges()
        
    def calculate_center(self, control_points: list[Point]) -> list[Point]:
        """Calculates the center line of the lane based on the given control points.
        
        Args:
            control_points (list[Point]): List of Point object with x, y values to create spline through.
        
        Returns:
            list[Point]: List of Point object representing the center line of the lane.
        """
        # Placeholder for actual implementation
        return control_points
    
    def calculate_edges(self) -> tuple[list[Point], list[Point]]:
        """Calculates the left and right edges of the lane based on the center line and lane width.
        Edges are calculated by offsetting the center line points by half the lane width in the perpendicular direction.
        The perpendicular direction is determined by the tangent of the spline at each point.
        The left edge is offset to the left of the center line and the right edge is offset to the right.
        
        Returns:
            tuple[list[Point], list[Point]]: Tuple of two lists of Point object representing the left and right edges of the lane.
        """
        left_edge = []
        right_edge = []
        # Placeholder for actual implementation
        return left_edge, right_edge
        