class NumberTracer:

        def __init__(self, coordinates):
            self.__x = coordinates[0]
            self.__y = coordinates[1]
            self.__w = coordinates[2]
            self.__h = coordinates[3]
            self.__blue = False
            self.__green = False
            self.__frameNumber = -1
            self.__value = -1
            self.__evidence=False

        def set_coordinates(self, coordinates):
            self.__x = coordinates[0]
            self.__y = coordinates[1]
            self.__w = coordinates[2]
            self.__h = coordinates[3]

        def get_coordinates(self):
            coordinates=[self.__x, self.__y, self.__w, self.__h]
            return coordinates

        def get_center(self):
            return self.__x + self.__w/2, self.__y + self.__h/2

        def get_bottom_left(self):
            return (self.__x, self.__y + self.__h)

        def get_bottom_right(self):
            return (self.__x + self.__w, self.__y + self.__h)

        def get_blue(self):
            return self.__blue

        def set_blue(self, crossed):
            self.__blue = crossed

        def get_green(self):
            return self.__green

        def set_green(self, crossed):
            self.__green = crossed

        def set_value(self, num_value):
            self.__value=num_value
        
        def get_value(self):
            return self.__value

        def set_frameNumber(self, frameNum):
            self.__frameNumber=frameNum
        
        def get_frameNumber(self):
            return self.__frameNumber

        def get_x(self):
            return self.__x

        def get_y(self):
            return self.__y

        def get_h(self):
            return self.__h

        def get_w(self):
            return self.__w

        def get_evidence(self):
            return self.__evidence

        def set_evidence(self, crossed):
            self.__evidence = crossed
        