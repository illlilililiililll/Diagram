import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt

class Diagram():
    def __init__(self, path):
        self.img = cv2.imread(path)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.rect = (0, 0, self.img.shape[1], self.img.shape[0])
        self.subdiv = cv2.Subdiv2D(self.rect)

    def Landmarks(self, show=True, imgshow=True):
        faces = self.detector(self.gray)
        landmarks = []

        for face in faces:
            shape = self.predictor(self.gray, face)
            for i in range(68):
                landmarks.append((shape.part(i).x, shape.part(i).y))
            
        self.landmarks = np.array(landmarks)

        if show:
            image = self.img.copy() if imgshow else np.ones_like(self.img) * 255

            for (x, y) in landmarks:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
                 
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

            plt.title('Facial Landmarks')
            plt.show()

    def Triangluation(self, imgshow=True, circle=False):
        # Insert landmarks into subdiv
        for point in self.landmarks:
            self.subdiv.insert((float(point[0]), float(point[1])))

        # Get Delaunay triangles
        triangles = self.subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)

        delaunay_img = self.img.copy() if imgshow else np.ones_like(self.img) * 255

        for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            cv2.line(delaunay_img, pt1, pt2, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            cv2.line(delaunay_img, pt2, pt3, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            cv2.line(delaunay_img, pt3, pt1, (0, 0, 0), 1, lineType=cv2.LINE_AA)

            if circle:
                x1, y1 = pt1
                x2, y2 = pt2
                x3, y3 = pt3
                
                mid1 = ((x1 + x2) / 2, (y1 + y2) / 2)
                mid2 = ((x2 + x3) / 2, (y2 + y3) / 2)
                
                slope1 = None if x2 - x1 == 0 else (y2 - y1) / (x2 - x1)
                slope2 = None if x3 - x2 == 0 else (y3 - y2) / (x3 - x2)
                
                # 수직이등분선의 기울기
                perp_slope1 = None if slope1 is None else -1 / slope1
                perp_slope2 = None if slope2 is None else -1 / slope2
                
                # 외접원의 중심
                if perp_slope1 is None:
                    center_x = mid1[0]
                    center_y = perp_slope2 * (center_x - mid2[0]) + mid2[1]
                elif perp_slope2 is None:
                    center_x = mid2[0]
                    center_y = perp_slope1 * (center_x - mid1[0]) + mid1[1]
                else:
                    center_x = (perp_slope1 * mid1[0] - perp_slope2 * mid2[0] + mid2[1] - mid1[1]) / (perp_slope1 - perp_slope2)
                    center_y = perp_slope1 * (center_x - mid1[0]) + mid1[1]
                
                # 반지름
                radius = ((center_x - x1) ** 2 + (center_y - y1) ** 2)**(1/2)

                try:
                    center_x = round(center_x)
                    center_y = round(center_y)
                    radius = round(radius)

                    import random
                    cv2.circle(delaunay_img, (center_x, center_y), radius, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 1, lineType=cv2.LINE_AA)
                
                except ValueError:
                    pass

        for point in self.landmarks:
            cv2.circle(delaunay_img, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
        
        plt.imshow(cv2.cvtColor(delaunay_img, cv2.COLOR_BGR2RGB))
        plt.xticks([]), plt.yticks([])
        plt.title('Delaunay Triangulation')
        plt.show()
    
    def Voronoi(self, imgshow=True):
        (facets, centers) = self.subdiv.getVoronoiFacetList([])

        voronoi_img = self.img.copy() if imgshow else np.ones_like(self.img) * 255
        
        for i in range(len(facets)):
            ifacet_arr = []
            for f in facets[i]:
                ifacet_arr.append(f)

            ifacet = np.array(ifacet_arr, np.int32)
            ifacets = np.array([ifacet])
            cv2.polylines(voronoi_img, ifacets, True, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            cv2.circle(voronoi_img, (int(centers[i][0]), int(centers[i][1])), 2, (0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)

        # Plot Voronoi diagram
        plt.imshow(cv2.cvtColor(voronoi_img, cv2.COLOR_BGR2RGB))
        plt.title('Voronoi Diagram')
        plt.xticks([])
        plt.yticks([])

        plt.show()

if __name__ == "__main__":
    path = 'img_path'

    D = Diagram(path)

    D.Landmarks()
    D.Triangluation()
    D.Voronoi()