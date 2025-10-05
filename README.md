# Bender-et-al-MicroFabrication
A repository containing the MATLAB code, CAD files, and detailed manufacturing procedures for the SLA micropatterning paper by Bender et al (Hsu Lab)

CAD Files
  Stripes - 200, 400, 600, 800            :    STL file for the striped-patterned mold with stripes of 200um, 400um, 600um, and 800um width, respectively
  Checkered Pattern - 200, 400, 600, 800  :    STL file for the checkered-patterned mold with the length of the squares being 200um, 400um, 600um, and 800um, respectively

MATLAB Files
  COA.m   :    MATLAB script that takes a DAPI image, applies a watershed mask, segments, and draws the ellipses, as well as extracts the centroid and orientation values of each ellipse
