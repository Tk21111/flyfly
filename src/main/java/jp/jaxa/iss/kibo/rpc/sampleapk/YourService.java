package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;
import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.RectF;

import org.opencv.android.Utils;
import org.opencv.aruco.Aruco;
import org.opencv.aruco.Dictionary;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point3;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Scalar;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;


/**
 * Main service class to run Kibo RPC mission plans.
 * This version includes refactored computer vision logic for improved
 * efficiency and robustness.
 */
public class YourService extends KiboRpcService {

    private TfLiteObjectDetectionHelper detector;
    private Map<Integer, AreaDetectionSummary> areaSummaries = new HashMap<>(); // Global map to store summaries per area

    // Helper class to store summary for each area (1-4)
    private static class AreaDetectionSummary {
        String bestDetectedLabel = "unknown";
        float bestDetectedScore = -1.0f;
        Point markerWorldPosition = null; // Storing the marker's 3D position
        int totalDetectionsInArea = 0;
    }

    // A simple wrapper class for returning a position and a boolean flag.
    // This is used by the triGeomatic method.
    public class PositionAndNegative {
        public Point position;
        public boolean negative;

        public PositionAndNegative(Point pos, boolean neg) {
            this.position = pos;
            this.negative = neg;
        }
    }

    // A simple wrapper class for returning a position and an orientation.
    public class PositionAndOrientation {
        public Point position;
        public Quaternion orientation;

        public PositionAndOrientation(Point pos, Quaternion quat) {
            this.position = pos;
            this.orientation = quat;
        }
    }


    @Override
    protected void runPlan1() {
        api.startMission();
        Log.i("Mission", "Starting runPlan1");

        // Initialize camera intrinsics and distortion coefficients
        Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
        cameraMatrix.put(0, 0, api.getNavCamIntrinsics()[0]);
        Mat cameraCoefficients = new Mat(1, 5, CvType.CV_64F);
        cameraCoefficients.put(0, 0, api.getNavCamIntrinsics()[1]);

        // Initialize TensorFlow Lite detection
        initializeDetection("best_with_metadata.tflite");

        // ArUco dictionary for marker detection
        Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);

        try {
            // PHASE 1: Navigate to first waypoint and detect objects for AREA 1
            Log.i("Mission", "Phase 1: Moving to first waypoint (Area 1)");
            Point point1 = new Point(10.9d, -9.92284d, 5.195d);
            Quaternion quaternion1 = new Quaternion(0f, 0f, -0.707f, 0.707f);
            api.moveTo(point1, quaternion1, false);

            Mat undistortedImage1 = undistortCam(api.getMatNavCam(), cameraMatrix, cameraCoefficients);
            api.saveMatImage(undistortedImage1, "waypoint1_undistorted.png");

            List<Mat> cornersList1 = new ArrayList<>();
            Mat ids1 = new Mat();
            Aruco.detectMarkers(undistortedImage1, dictionary, cornersList1, ids1);

            Point[] markerPositions1 = arToPos(cornersList1, cameraMatrix, cameraCoefficients, 0.05f); // Get marker positions

            List<Mat> croppedPatches1 = arToPaperAndCrop(cornersList1, undistortedImage1);
            Log.i("Mission", "Found " + croppedPatches1.size() + " cropped regions at waypoint 1 for Area 1");

            // Initialize summary for Area 1
            AreaDetectionSummary area1Summary = new AreaDetectionSummary();
            if (markerPositions1.length > 0 && markerPositions1[0] != null) {
                area1Summary.markerWorldPosition = markerPositions1[0]; // Assuming first marker is primary for the area
            }

            List<TfLiteObjectDetectionHelper.DetectionResult> allDetectionsInArea1 = new ArrayList<>();
            for (int i = 0; i < croppedPatches1.size(); i++) {
                List<TfLiteObjectDetectionHelper.DetectionResult> currentPatchDetections = getPatchDetections(croppedPatches1.get(i), "Area1_Crop" + i);
                allDetectionsInArea1.addAll(currentPatchDetections);
                api.saveMatImage(croppedPatches1.get(i), "Area1_Crop" + i + ".png");
                croppedPatches1.get(i).release();
            }
            undistortedImage1.release();

            // Process and store summary for Area 1
            processAndStoreAreaSummary(1, allDetectionsInArea1, area1Summary);


            // PHASE 2: Navigate to second waypoint and detect objects for AREA 2
            Log.i("Mission", "Phase 2: Moving to second waypoint (Area 2)");
            Point point2 = new Point(10.6d, -8.0484d, 4.5d); // Updated Y-coordinate to avoid hardcoded 8.5484 for a potential mission specific point
            Quaternion quaternion2 = toQuaternion(0, Math.toRadians(90), Math.toRadians(90));
            api.moveTo(point2, quaternion2, false);

            Mat undistortedImage2 = undistortCam(api.getMatNavCam(), cameraMatrix, cameraCoefficients);
            api.saveMatImage(undistortedImage2, "waypoint2_undistorted.png");

            List<Mat> cornersList2 = new ArrayList<>();
            Mat ids2 = new Mat();
            Aruco.detectMarkers(undistortedImage2, dictionary, cornersList2, ids2);

            Point[] markerPositions2 = arToPos(cornersList2, cameraMatrix, cameraCoefficients, 0.05f);

            List<Mat> croppedPatches2 = arToPaperAndCrop(cornersList2, undistortedImage2);
            Log.i("Mission", "Found " + croppedPatches2.size() + " cropped regions at waypoint 2 for Area 2");

            AreaDetectionSummary area2Summary = new AreaDetectionSummary();
            if (markerPositions2.length > 0 && markerPositions2[0] != null) {
                area2Summary.markerWorldPosition = markerPositions2[0];
            }

            List<TfLiteObjectDetectionHelper.DetectionResult> allDetectionsInArea2 = new ArrayList<>();
            for (int i = 0; i < croppedPatches2.size(); i++) {
                List<TfLiteObjectDetectionHelper.DetectionResult> currentPatchDetections = getPatchDetections(croppedPatches2.get(i), "Area2_Crop" + i);
                allDetectionsInArea2.addAll(currentPatchDetections);
                api.saveMatImage(croppedPatches2.get(i), "Area2_Crop" + i + ".png");
                croppedPatches2.get(i).release();
            }
            undistortedImage2.release();

            processAndStoreAreaSummary(2, allDetectionsInArea2, area2Summary);


            // PHASE 3: Navigate to third waypoint and detect objects for AREA 3
            Log.i("Mission", "Phase 3: Moving to third waypoint (Area 3)");
            Point point3 = new Point(10.6d, -7.8484d, 4.5d);
            Quaternion quaternion3 = toQuaternion(0, Math.toRadians(90), Math.toRadians(90));
            api.moveTo(point3, quaternion3, false);

            Mat undistortedImage3 = undistortCam(api.getMatNavCam(), cameraMatrix, cameraCoefficients);
            api.saveMatImage(undistortedImage3, "waypoint3_undistorted.png");

            List<Mat> cornersList3 = new ArrayList<>();
            Mat ids3 = new Mat();
            Aruco.detectMarkers(undistortedImage3, dictionary, cornersList3, ids3); // Correctly detect markers for Area 3

            Point[] markerPositions3 = arToPos(cornersList3, cameraMatrix, cameraCoefficients, 0.05f);

            List<Mat> croppedPatches3 = arToPaperAndCrop(cornersList3, undistortedImage3);
            Log.i("Mission", "Found " + croppedPatches3.size() + " cropped regions at waypoint 3 for Area 3");

            AreaDetectionSummary area3Summary = new AreaDetectionSummary();
            if (markerPositions3.length > 0 && markerPositions3[0] != null) {
                area3Summary.markerWorldPosition = markerPositions3[0];
            }

            List<TfLiteObjectDetectionHelper.DetectionResult> allDetectionsInArea3 = new ArrayList<>();
            for (int i = 0; i < croppedPatches3.size(); i++) {
                List<TfLiteObjectDetectionHelper.DetectionResult> currentPatchDetections = getPatchDetections(croppedPatches3.get(i), "Area3_Crop" + i);
                allDetectionsInArea3.addAll(currentPatchDetections);
                api.saveMatImage(croppedPatches3.get(i), "Area3_Crop" + i + ".png");
                croppedPatches3.get(i).release();
            }
            undistortedImage3.release();

            processAndStoreAreaSummary(3, allDetectionsInArea3, area3Summary);


            // PHASE 4: Navigate to fourth waypoint and detect objects for AREA 4
            Log.i("Mission", "Phase 4: Moving to fourth waypoint (Area 4)");
            Point point4 = new Point(10.9d, -7.0d, 4.7d);
            Quaternion quaternion4 = new Quaternion(0f, 0f, 1f, 0f);
            api.moveTo(point4, quaternion4, false);

            Mat undistortedImage4 = undistortCam(api.getMatNavCam(), cameraMatrix, cameraCoefficients);
            api.saveMatImage(undistortedImage4, "waypoint4_undistorted.png");

            List<Mat> cornersList4 = new ArrayList<>();
            Mat ids4 = new Mat();
            Aruco.detectMarkers(undistortedImage4, dictionary, cornersList4, ids4); // Correctly detect markers for Area 4

            Point[] markerPositions4 = arToPos(cornersList4, cameraMatrix, cameraCoefficients, 0.05f);

            List<Mat> croppedPatches4 = arToPaperAndCrop(cornersList4, undistortedImage4);
            Log.i("Mission", "Found " + croppedPatches4.size() + " cropped regions at waypoint 4 for Area 4");

            AreaDetectionSummary area4Summary = new AreaDetectionSummary();
            if (markerPositions4.length > 0 && markerPositions4[0] != null) {
                area4Summary.markerWorldPosition = markerPositions4[0];
            }

            List<TfLiteObjectDetectionHelper.DetectionResult> allDetectionsInArea4 = new ArrayList<>();
            for (int i = 0; i < croppedPatches4.size(); i++) {
                List<TfLiteObjectDetectionHelper.DetectionResult> currentPatchDetections = getPatchDetections(croppedPatches4.get(i), "Area4_Crop" + i);
                allDetectionsInArea4.addAll(currentPatchDetections);
                api.saveMatImage(croppedPatches4.get(i), "Area4_Crop" + i + ".png");
                croppedPatches4.get(i).release();
            }
            undistortedImage4.release();

            processAndStoreAreaSummary(4, allDetectionsInArea4, area4Summary);


            // PHASE 5: Move in front of astronaut and report completion
            Log.i("Mission", "Phase 5: Moving to astronaut position");
            Point astronautPoint = new Point(11.143d, -6.7607d, 4.9654d);
            Quaternion astronautQuaternion = new Quaternion(0f, 0f, 0.707f, 0.707f);
            api.moveTo(astronautPoint, astronautQuaternion, false);
            api.reportRoundingCompletion();

            // PHASE 6: Final positioning and target snapshot - Move to area with specific item
            Log.i("Mission", "Phase 6: Final target positioning - Searching for target item.");
            api.notifyRecognitionItem(); // Notify recognition item before final move/snapshot

            // Define your target item. Example: "Treasure_box" or a specific gem.
            // You might want to get this from a mission parameter if available.
            String targetItemToFind = "Treasure_box"; // Change this to the actual item you need to find.
            // Or if you are looking for any gem:
            // String targetItemToFind = "Crystal"; // Or "Diamond", "Emerald" based on mission

            Point finalTargetPosition = null;
            Quaternion finalTargetOrientation = null;
            int foundAreaId = -1;

            // Iterate through the summaries to find the target item
            for (Map.Entry<Integer, AreaDetectionSummary> entry : areaSummaries.entrySet()) {
                int currentAreaId = entry.getKey();
                AreaDetectionSummary summary = entry.getValue();

                if (targetItemToFind.equals(summary.bestDetectedLabel)) {
                    // Found the target item in this area
                    finalTargetPosition = summary.markerWorldPosition;
                    foundAreaId = currentAreaId;
                    Log.i("Mission", "Target item '" + targetItemToFind + "' found in Area " + foundAreaId);
                    break; // Found it, no need to search further
                }
                // Also check for gems if the primary target wasn't found immediately
                if (("Crystal".equals(targetItemToFind) || "Diamond".equals(targetItemToFind) || "Emerald".equals(targetItemToFind))) {
                    if (("Crystal".equals(summary.bestDetectedLabel) || "Diamond".equals(summary.bestDetectedLabel) || "Emerald".equals(summary.bestDetectedLabel))) {
                        // If any gem is considered the target, and we found one.
                        finalTargetPosition = summary.markerWorldPosition;
                        foundAreaId = currentAreaId;
                        Log.i("Mission", "A gem ('" + summary.bestDetectedLabel + "') found in Area " + foundAreaId + " as target.");
                        break;
                    }
                }
            }

            // Move to the identified target position or a safe default
            if (finalTargetPosition != null) {
                // Calculate orientation to face the target marker and move to 30cm away
                PositionAndOrientation targetPose = calculateMarker30PositionAndOrientation(finalTargetPosition);
                Log.i("Mission", "Calculated final target position: " + targetPose.position);
                Log.i("Mission", "Calculated final target orientation: " + targetPose.orientation);

                if (isValidPosition(targetPose.position)) {
                    api.moveTo(targetPose.position, targetPose.orientation, false);
                } else {
                    Log.w("Mission", "Calculated final position out of bounds, using safe position. Moving to astronaut position.");
                    Point safePos = new Point(11.143, -6.7607, 4.9654); // Astronaut position as safe fallback
                    api.moveTo(safePos, astronautQuaternion, false);
                }
            } else {
                Log.w("Mission", "Target item '" + targetItemToFind + "' not found in any area. Moving to astronaut position as fallback.");
                Point safePos = new Point(11.143, -6.7607, 4.9654); // Astronaut position as safe fallback
                api.moveTo(safePos, astronautQuaternion, false);
            }

            api.takeTargetItemSnapshot(); // Take snapshot after moving to the final target
            Log.i("Mission", "Target item snapshot completed");
            // The finalImage was taken right after api.notifyRecognitionItem();, release it here:
            // finalImage.release(); // if you comment the line above, this release needs to be inside the try block for the final image.

        } catch (Exception e) {
            Log.e("Mission", "Error during mission execution", e);
        } finally {
            if (detector != null) {
                detector.close();
            }
            cameraMatrix.release();
            cameraCoefficients.release();
            Log.i("Mission", "runPlan1 finished and resources released.");
        }
    }

    @Override
    protected void runPlan2() { /* ... */ }
    @Override
    protected void runPlan3() { /* ... */ }

    /**
     * Helper to initialize the TFLite Object Detector.
     */
    public void initializeDetection(String modelFilename) {
        try {
            List<String> labels = loadLabels(getApplicationContext(), "labels.txt");
            detector = new TfLiteObjectDetectionHelper(getApplicationContext(), modelFilename, labels);
            Log.i("DetectionInit", "TFLite model loaded successfully: " + modelFilename);
        } catch (IOException e) {
            Log.e("DetectionInit", "Failed to load TFLite model: " + modelFilename, e);
            detector = null;
        }
    }

    // Add a method to load labels from your assets folder
    private List<String> loadLabels(Context context, String labelPath) throws IOException {
        List<String> labels = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(context.getAssets().open(labelPath)))) {
            String line;
            while ((line = reader.readLine()) != null) {
                labels.add(line);
            }
        }
        return labels;
    }


    /**
     * Undistorts a camera image using pre-supplied camera intrinsics.
     */
    private Mat undistortCam(Mat image, Mat cameraMatrix, Mat cameraCoefficients) {
        Mat undistorted = new Mat();
        Calib3d.undistort(image, undistorted, cameraMatrix, cameraCoefficients);
        return undistorted;
    }

    /**
     * Converts Euler angles (in radians) to a Quaternion.
     */
    private Quaternion toQuaternion(double roll, double pitch, double yaw) {
        double cy = Math.cos(yaw * 0.5);
        double sy = Math.sin(yaw * 0.5);
        double cp = Math.cos(pitch * 0.5);
        double sp = Math.sin(pitch * 0.5);
        double cr = Math.cos(roll * 0.5);
        double sr = Math.sin(roll * 0.5);

        float w = (float) (cr * cp * cy + sr * sp * sy);
        float x = (float) (sr * cp * cy - cr * sp * sy);
        float y = (float) (cr * sp * cy + sr * cp * sy);
        float z = (float) (cr * cp * sy - sr * sp * cy);

        return new Quaternion(x, y, z, w);
    }

    /**
     * Takes a list of ArUco marker corners and an image, and returns a list of
     * straightened, cropped images of the areas of interest next to each marker.
     * This version uses an efficient perspective warp.
     */
    public List<Mat> arToPaperAndCrop(List<Mat> cornersList, Mat sourceImage) {
        if (cornersList == null || cornersList.isEmpty()) {
            Log.w("arToPaperAndCrop", "No ArUco markers found to crop.");
            return new ArrayList<>();
        }

        List<Mat> croppedRegions = new ArrayList<>();
        final double PAPER_WIDTH_CM = 26.0;
        final double PAPER_HEIGHT_CM = 22.0;
        final double MARKER_SIZE_CM = 5.0;
        final double OFFSET_X_CM = -26.0; // Paper is to the left of the marker
        final double OFFSET_Y_CM = -6.0; // Paper is above the marker

        for (Mat corners : cornersList) {
            if (corners.empty() || corners.rows() == 0) continue;

            // ArUco corners are ordered: top-left, top-right, bottom-right, bottom-left
            org.opencv.core.Point tl = new org.opencv.core.Point(corners.get(0, 0));
            org.opencv.core.Point tr = new org.opencv.core.Point(corners.get(0, 1));

            // Calculate pixels per centimeter based on the top edge of the marker
            double dx = tr.x - tl.x;
            double dy = tr.y - tl.y;
            double topEdgePixelLength = Math.sqrt(dx * dx + dy * dy);
            double pixelsPerCm = topEdgePixelLength / MARKER_SIZE_CM;

            // Define the coordinate system relative to the top-left corner of the marker
            // X-axis along the top edge, Y-axis perpendicular to it
            double angle = Math.atan2(dy, dx);
            double cosA = Math.cos(angle);
            double sinA = Math.sin(angle);

            // Calculate the four corners of the paper in the image's coordinate system
            // Start from the marker's top-left corner and apply offsets and rotations
            org.opencv.core.Point paper_tl = new org.opencv.core.Point(
                    tl.x + (OFFSET_X_CM * cosA - OFFSET_Y_CM * sinA) * pixelsPerCm,
                    tl.y + (OFFSET_X_CM * sinA + OFFSET_Y_CM * cosA) * pixelsPerCm
            );
            org.opencv.core.Point paper_tr = new org.opencv.core.Point(
                    paper_tl.x + (PAPER_WIDTH_CM * cosA) * pixelsPerCm,
                    paper_tl.y + (PAPER_WIDTH_CM * sinA) * pixelsPerCm
            );
            org.opencv.core.Point paper_bl = new org.opencv.core.Point(
                    paper_tl.x - (PAPER_HEIGHT_CM * sinA) * pixelsPerCm,
                    paper_tl.y + (PAPER_HEIGHT_CM * cosA) * pixelsPerCm
            );
            org.opencv.core.Point paper_br = new org.opencv.core.Point(
                    paper_bl.x + (PAPER_WIDTH_CM * cosA) * pixelsPerCm,
                    paper_bl.y + (PAPER_WIDTH_CM * sinA) * pixelsPerCm
            );

            // Define the source and destination points for the perspective transform
            MatOfPoint2f srcPoints = new MatOfPoint2f(paper_tl, paper_tr, paper_br, paper_bl);

            // Output image size
            Size outSize = new Size(PAPER_WIDTH_CM * pixelsPerCm, PAPER_HEIGHT_CM * pixelsPerCm);
            MatOfPoint2f dstPoints = new MatOfPoint2f(
                    new org.opencv.core.Point(0, 0),
                    new org.opencv.core.Point(outSize.width - 1, 0),
                    new org.opencv.core.Point(outSize.width - 1, outSize.height - 1),
                    new org.opencv.core.Point(0, outSize.height - 1)
            );

            // Get the perspective transform matrix and apply it
            Mat transform = Imgproc.getPerspectiveTransform(srcPoints, dstPoints);
            Mat cropped = new Mat();
            Imgproc.warpPerspective(sourceImage, cropped, transform, outSize);

            croppedRegions.add(cropped);

            // Release intermediate mats
            srcPoints.release();
            dstPoints.release();
            transform.release();
        }

        return croppedRegions;
    }

    /**
     * Estimates the 3D world position of detected ArUco markers.
     */
    private Point[] arToPos(List<Mat> cornersList, Mat cameraMatrix, Mat distCoeffs, float markerLength) {
        Point[] positions = new Point[cornersList.size()];
        Mat rvecs = new Mat();
        Mat tvecs = new Mat();

        // Use estimatePoseSingleMarkers for robustness
        Aruco.estimatePoseSingleMarkers(cornersList, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);

        for (int i = 0; i < cornersList.size(); i++) {
            double[] tvecArr = new double[3];
            tvecs.get(i, 0, tvecArr); // tvec = [X_cam, Y_cam, Z_cam]

            // Get robot's current position and orientation in the world frame
            gov.nasa.arc.astrobee.Kinematics kin = api.getRobotKinematics();
            Point camPosWorld = kin.getPosition();
            Quaternion camOrientation = kin.getOrientation();

            // Convert camera-frame marker translation to a world-frame vector
            Point tvecInCamFrame = new Point(tvecArr[0], tvecArr[1], tvecArr[2]);
            Point tvecInWorldFrame = quatRotate(camOrientation, tvecInCamFrame);

            // Add the robot's world position to get the marker's world position
            Point worldPos = new Point(
                    camPosWorld.getX() + tvecInWorldFrame.getX(),
                    camPosWorld.getY() + tvecInWorldFrame.getY(),
                    camPosWorld.getZ() + tvecInWorldFrame.getZ()
            );

            Log.i("arToPos", "Marker " + i + " world position: " + worldPos);
            positions[i] = worldPos;
        }

        rvecs.release();
        tvecs.release();
        return positions;
    }

    /**
     * Rotates a point by a quaternion.
     */
    private Point quatRotate(Quaternion q, Point v) {
        // q*v
        double nw = -q.getX() * v.getX() - q.getY() * v.getY() - q.getZ() * v.getZ();
        double nx = q.getW() * v.getX() + q.getY() * v.getZ() - q.getZ() * v.getY();
        double ny = q.getW() * v.getY() - q.getX() * v.getZ() + q.getZ() * v.getX();
        double nz = q.getW() * v.getZ() + q.getX() * v.getY() - q.getY() * v.getX();
        Quaternion v_quat = new Quaternion((float)nx, (float)ny, (float)nz, (float)nw);

        // (q*v)*q_inv
        // q_inv is (-x, -y, -z, w) for a unit quaternion
        double final_x = v_quat.getW() * -q.getX() + v_quat.getX() * q.getW() + v_quat.getY() * -q.getZ() - v_quat.getZ() * -q.getY();
        double final_y = v_quat.getW() * -q.getY() - v_quat.getX() * -q.getZ() + v_quat.getY() * q.getW() + v_quat.getZ() * -q.getX();
        double final_z = v_quat.getW() * -q.getZ() + v_quat.getX() * -q.getY() - v_quat.getY() * -q.getX() + v_quat.getZ() * q.getW();

        return new Point(final_x, final_y, final_z);
    }

    /**
     * Placeholder implementation for the geometric calculation to find a target position.
     * This calculates a point 30cm away from the target point along the line connecting the robot and the target.
     */
    private PositionAndNegative triGeomatic(double p1_robot, double p2_robot, double p1_marker, double p2_marker, double p3_plane) {
        final double OFFSET_DISTANCE = 0.30; // 30 cm
        double d1 = p1_marker - p1_robot;
        double d2 = p2_marker - p2_robot;
        double current_dist = Math.sqrt(d1 * d1 + d2 * d2);

        // Calculate the unit vector from robot to marker
        double u1 = d1 / current_dist;
        double u2 = d2 / current_dist;

        // Calculate the target point by stepping back from the marker along the unit vector
        double target_p1 = p1_marker - u1 * OFFSET_DISTANCE;
        double target_p2 = p2_marker - u2 * OFFSET_DISTANCE;

        Point finalPos = new Point(target_p1, target_p2, p3_plane);
        boolean isNegative = (d1 + d2 < 0); // Example condition

        return new PositionAndNegative(finalPos, isNegative);
    }

    /**
     * Calculates the final target position and orientation based on the marker's location.
     */
    private PositionAndOrientation calculateMarker30PositionAndOrientation(Point markerPos) {
        gov.nasa.arc.astrobee.Kinematics currKinematics = api.getRobotKinematics();
        Point currPos = currKinematics.getPosition();
        Point toPos;
        Quaternion r;

        Log.i("calculateMarker30", "Current pos: " + currPos + ", Target marker pos: " + markerPos);

        // Logic based on the marker's location in the ISS module
        // Note: These conditions should be refined based on the actual mission map and target areas.
        if (markerPos.getY() < -9.5) { // Assuming Y < -9.5 is roughly "Area 1" (Air lock area)
            Log.i("calculateMarker30", "Calculating for air lock area (XY plane)");
            PositionAndNegative tmp = triGeomatic(currPos.getX(), currPos.getY(), markerPos.getX(), markerPos.getY(), markerPos.getZ());
            toPos = tmp.position;
            // Face the marker from 30cm back (yaw from robot to marker)
            double yaw = Math.atan2(markerPos.getY() - toPos.getY(), markerPos.getX() - toPos.getX());
            r = toQuaternion(0, 0, yaw); // Roll 0, Pitch 0

        } else if (markerPos.getZ() < 4.8 && markerPos.getY() > -9.5) { // Assuming Z < 4.8 and Y > -9.5 is "Area 2/3" (Top area looking down)
            Log.i("calculateMarker30", "Calculating for top area (ZY plane)");
            // For ZY plane, triGeomatic should operate on (current_Z, current_Y) to (marker_Z, marker_Y)
            PositionAndNegative tmp = triGeomatic(currPos.getZ(), currPos.getY(), markerPos.getZ(), markerPos.getY(), markerPos.getX());
            // The result from triGeomatic is in (z,y) order, with original_X as the fixed plane.
            // Remap back to (x,y,z) for Point: new Point(fixed_X, result_Y, result_Z)
            toPos = new Point(tmp.position.getZ(), tmp.position.getY(), tmp.position.getX()); // z,y,x from triGeomatic becomes x,y,z here
            r = toQuaternion(0, Math.toRadians(-90), Math.toRadians(180)); // Look down
        }
        else if (markerPos.getX() < 10.0 && markerPos.getY() > -9.5 && markerPos.getZ() > 4.8) { // Assuming X < 10.0 is "Area 4" (Side area)
            Log.i("calculateMarker30", "Calculating for side area (XY plane)");
            PositionAndNegative tmp = triGeomatic(currPos.getX(), currPos.getY(), markerPos.getX(), markerPos.getY(), markerPos.getZ());
            toPos = tmp.position;
            double yaw = Math.atan2(markerPos.getY() - toPos.getY(), markerPos.getX() - toPos.getX());
            r = toQuaternion(0, 0, yaw);
        }
        else { // Default or general approach for other areas
            Log.i("calculateMarker30", "Using default approach calculation (general XY plane)");
            PositionAndNegative tmp = triGeomatic(currPos.getX(), currPos.getY(), markerPos.getX(), markerPos.getY(), markerPos.getZ());
            toPos = tmp.position;
            double yaw = Math.atan2(markerPos.getY() - toPos.getY(), markerPos.getX() - toPos.getX());
            r = toQuaternion(0, 0, yaw);
        }

        Log.i("calculateMarker30", "Calculated target pose -> Pos: " + toPos + " | Quat: " + r);
        return new PositionAndOrientation(toPos, r);
    }

    /**
     * Checks if a target position is within the valid Kibo module operational zone.
     */
    private boolean isValidPosition(Point pos) {
        // Approximate bounds of the Kibo module. Adjust as needed.
        double minX = 9.5,  maxX = 11.5;
        double minY = -10.5, maxY = -6.0;
        double minZ = 4.0,  maxZ = 6.0;

        boolean inBounds = pos.getX() >= minX && pos.getX() <= maxX &&
                pos.getY() >= minY && pos.getY() <= maxY &&
                pos.getZ() >= minZ && pos.getZ() <= maxZ;

        if (!inBounds) {
            Log.e("isValidPosition", "Position " + pos + " is out of bounds!");
        }
        return inBounds;
    }

    /**
     * Helper method to process a list of detections for an area and store the summary.
     * This method also calls api.setAreaInfo.
     * @param areaNumber The ID of the area (1-4).
     * @param allDetections The aggregated list of all detections from all patches in this area.
     * @param areaSummary The AreaDetectionSummary object to update and store.
     */
    private void processAndStoreAreaSummary(int areaNumber,
                                            List<TfLiteObjectDetectionHelper.DetectionResult> allDetections,
                                            AreaDetectionSummary areaSummary) {

        areaSummary.totalDetectionsInArea = allDetections.size();

        String bestAreaLabel = "unknown";
        float bestAreaScore = -1.0f;

        // Find the best gem or overall best item for this Area
        for (TfLiteObjectDetectionHelper.DetectionResult detection : allDetections) {
            String currentLabel = detection.label;
            // Prioritize "Crystal", "Diamond", "Emerald"
            if ("Crystal".equals(currentLabel) || "Diamond".equals(currentLabel) || "Emerald".equals(currentLabel)) {
                if (detection.score > bestAreaScore) {
                    bestAreaScore = detection.score;
                    bestAreaLabel = currentLabel;
                }
            }
        }

        if (bestAreaLabel != null && !bestAreaLabel.equals("unknown")) {
            areaSummary.bestDetectedLabel = bestAreaLabel;
            areaSummary.bestDetectedScore = bestAreaScore;
        } else if (!allDetections.isEmpty()) {
            // If no specific gem, take the overall best score (list is already sorted by score)
            areaSummary.bestDetectedLabel = allDetections.get(0).label;
            areaSummary.bestDetectedScore = allDetections.get(0).score;
        } else {
            // No detections at all
            areaSummary.bestDetectedLabel = "unknown";
            areaSummary.bestDetectedScore = -1.0f;
        }

        // Store the summary for this Area
        areaSummaries.put(areaNumber, areaSummary);

        // Call api.setAreaInfo for this area
        api.setAreaInfo(areaNumber, areaSummary.bestDetectedLabel, areaSummary.totalDetectionsInArea);
        Log.i("AreaInfo", "Set Area " + areaNumber + " to: " + areaSummary.bestDetectedLabel + ", Count: " + areaSummary.totalDetectionsInArea);
    }


    /**
     * Performs object detection on a cropped image patch and returns detected items.
     * This method does not set api.setAreaInfo directly.
     */
    private List<TfLiteObjectDetectionHelper.DetectionResult> getPatchDetections(Mat croppedMat, String logPrefix) {
        if (detector == null) {
            Log.e("Detection", logPrefix + " - Detector not initialized.");
            return new ArrayList<>();
        }
        if (croppedMat == null || croppedMat.empty()) {
            Log.w("Detection", logPrefix + " - Input image is empty.");
            return new ArrayList<>();
        }

        Bitmap croppedBitmap = null;
        List<TfLiteObjectDetectionHelper.DetectionResult> detections = new ArrayList<>();
        try {
            croppedBitmap = Bitmap.createBitmap(croppedMat.cols(), croppedMat.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(croppedMat, croppedBitmap);

            detections = detector.detectObjects(croppedBitmap);

            if (!detections.isEmpty()) {
                Log.i("Detection", logPrefix + " - Detections found: " + detections.size());
            } else {
                Log.w("Detection", logPrefix + " - No objects detected in patch.");
            }
        } catch (Exception e) {
            Log.e("Detection", logPrefix + " - Error during patch detection process", e);
        } finally {
            if (croppedBitmap != null && !croppedBitmap.isRecycled()) {
                croppedBitmap.recycle();
            }
        }
        return detections;
    }


    /**
     * Inner helper class for TensorFlow Lite object detection.
     * This class handles loading the model, running inference, and post-processing.
     * Making it static allows it to be instantiated without an enclosing YourService instance.
     */
    public static class TfLiteObjectDetectionHelper {
        private Interpreter interpreter;
        private final List<String> labels;
        private final ImageProcessor imageProcessor;

        // Model parameters (adjust if your model has different input/output sizes or normalization)
        private static final int MODEL_INPUT_SIZE = 640; // Common YOLOv8 input size
        private static final float INPUT_NORM_MEAN = 0.0f;
        private static final float INPUT_NORM_STD = 255.0f; // YOLOv8 typically expects [0, 1] input, so divide by 255
        private static final float SCORE_THRESHOLD = 0.25f; // Threshold for objectness score (or combined score)
        private static final float IOU_THRESHOLD = 0.45f;       // IoU threshold for NMS
        private static final int MAX_RESULTS = 100; // Max number of detections after NMS

        // Inner class to hold detection results
        public static class DetectionResult {
            public final RectF boundingBox;
            public final String label;
            public final float score; // combined confidence score

            public DetectionResult(RectF box, String label, float score) {
                this.boundingBox = box;
                this.label = label;
                this.score = score;
            }
        }

        public TfLiteObjectDetectionHelper(Context context, String modelFilename, List<String> labels) throws IOException {
            this.labels = labels;
            MappedByteBuffer tfliteModel = loadModelFile(context, modelFilename);
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(4); // You can adjust threads
            interpreter = new Interpreter(tfliteModel, options);

            // Image processor for consistent resizing and normalization
            imageProcessor = new ImageProcessor.Builder()
                    .add(new ResizeOp(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
                    .add(new NormalizeOp(INPUT_NORM_MEAN, INPUT_NORM_STD)) // Normalizes pixels from 0-255 to 0-1
                    .build();

            Log.i("TfLiteObjectDetectionHelper", "Model loaded. Input size: " + MODEL_INPUT_SIZE + "x" + MODEL_INPUT_SIZE + ", Num Classes: " + labels.size());
        }

        private MappedByteBuffer loadModelFile(Context context, String modelFilename) throws IOException {
            AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelFilename);
            try (FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
                FileChannel fileChannel = inputStream.getChannel();
                long startOffset = fileDescriptor.getStartOffset();
                long declaredLength = fileDescriptor.getDeclaredLength();
                return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
            }
        }

        public List<DetectionResult> detectObjects(Bitmap bitmap) {
            if (bitmap == null) {
                Log.w("TfLiteObjectDetectionHelper", "Input bitmap is null.");
                return new ArrayList<>();
            }

            // 1. Preprocess the input bitmap
            // Ensure bitmap is ARGB_8888 for TensorImage
            Bitmap argbBitmap = bitmap.getConfig() == Bitmap.Config.ARGB_8888 ? bitmap : bitmap.copy(Bitmap.Config.ARGB_8888, false);
            TensorImage inputImage = new TensorImage(DataType.FLOAT32);
            inputImage.load(argbBitmap);
            inputImage = imageProcessor.process(inputImage);

            // 2. Prepare the output buffer
            // Common YOLOv8 TFLite output shape is [1, num_boxes, 5 + num_classes]
            // where num_boxes is often 8400 for 640x640 input, and 5 is [cx, cy, w, h, objectness_score]
            // Make sure to confirm this shape using Netron or by logging interpreter.getOutputTensor(0).shape()
            int[] outputShape = interpreter.getOutputTensor(0).shape(); // e.g., [1, 8400, 18] for 13 classes
            int numBoxes = outputShape[1];
            int outputDataSize = outputShape[2]; // 5 + numClasses (e.g., 5 + 13 = 18)

            // TensorBuffer to receive output
            TensorBuffer outputTensorBuffer = TensorBuffer.createFixedSize(outputShape, DataType.FLOAT32);

            // Map for interpreter outputs
            java.util.Map<Integer, Object> outputMap = new java.util.HashMap<>();
            outputMap.put(0, outputTensorBuffer.getBuffer());

            // 3. Run inference
            interpreter.runForMultipleInputsOutputs(new Object[]{inputImage.getBuffer()}, outputMap);

            // 4. Post-process the output tensor
            List<DetectionResult> rawDetections = new ArrayList<>();

            // Rewind the buffer before reading
            outputTensorBuffer.getBuffer().rewind();
            for (int i = 0; i < numBoxes; i++) {
                float x_center = outputTensorBuffer.getBuffer().getFloat();
                float y_center = outputTensorBuffer.getBuffer().getFloat();
                float width = outputTensorBuffer.getBuffer().getFloat();
                float height = outputTensorBuffer.getBuffer().getFloat();
                float objectnessScore = outputTensorBuffer.getBuffer().getFloat(); // Objectness score is at index 4

                float maxClassScore = 0;
                int detectedClassId = -1;

                // Read class probabilities
                for (int j = 0; j < labels.size(); j++) {
                    float classScore = outputTensorBuffer.getBuffer().getFloat();
                    if (classScore > maxClassScore) {
                        maxClassScore = classScore;
                        detectedClassId = j;
                    }
                }

                // Combined confidence
                float finalScore = objectnessScore * maxClassScore;

                if (finalScore > SCORE_THRESHOLD) {
                    // Convert normalized coordinates to pixel coordinates
                    float originalWidth = (float) argbBitmap.getWidth();
                    float originalHeight = (float) argbBitmap.getHeight();

                    // Calculate bounding box in original image pixels
                    float left = (x_center - width / 2) * originalWidth / MODEL_INPUT_SIZE;
                    float top = (y_center - height / 2) * originalHeight / MODEL_INPUT_SIZE;
                    float right = (x_center + width / 2) * originalWidth / MODEL_INPUT_SIZE;
                    float bottom = (y_center + height / 2) * originalHeight / MODEL_INPUT_SIZE;

                    // Clamp bounding box coordinates to image bounds
                    left = Math.max(0f, left);
                    top = Math.max(0f, top);
                    right = Math.min(originalWidth, right);
                    bottom = Math.min(originalHeight, bottom);

                    RectF scaledBox = new RectF(left, top, right, bottom);
                    String label = labels.get(detectedClassId);
                    rawDetections.add(new DetectionResult(scaledBox, label, finalScore));
                }
            }

            // 5. Apply Non-Maximum Suppression (NMS)
            List<DetectionResult> nmsDetections = applyNMS(rawDetections, IOU_THRESHOLD);

            // Sort by confidence score (descending) and return top MAX_RESULTS
            Collections.sort(nmsDetections, new Comparator<DetectionResult>() {
                @Override
                public int compare(DetectionResult o1, DetectionResult o2) {
                    return Float.compare(o2.score, o1.score); // Descending order
                }
            });

            if (nmsDetections.size() > MAX_RESULTS) {
                return nmsDetections.subList(0, MAX_RESULTS);
            } else {
                return nmsDetections;
            }
        }

        // NMS helper function (adapted from previous responses)
        private List<DetectionResult> applyNMS(List<DetectionResult> detections, float iouThreshold) {
            // Sort detections by confidence score in descending order
            Collections.sort(detections, new Comparator<TfLiteObjectDetectionHelper.DetectionResult>() {
                @Override
                public int compare(TfLiteObjectDetectionHelper.DetectionResult d1, TfLiteObjectDetectionHelper.DetectionResult d2) {
                    return Float.compare(d2.score, d1.score); // Descending order
                }
            });

            List<DetectionResult> nmsDetections = new ArrayList<>();
            boolean[] suppressed = new boolean[detections.size()];

            for (int i = 0; i < detections.size(); i++) {
                if (suppressed[i]) {
                    continue;
                }

                DetectionResult currentDetection = detections.get(i);
                nmsDetections.add(currentDetection);

                for (int j = i + 1; j < detections.size(); j++) {
                    if (suppressed[j]) {
                        continue;
                    }

                    DetectionResult otherDetection = detections.get(j);

                    // Apply NMS per class (common practice)
                    if (currentDetection.label.equals(otherDetection.label)) { // Compare by label (class)
                        float iou = calculateIoU(currentDetection.boundingBox, otherDetection.boundingBox);
                        if (iou > iouThreshold) {
                            suppressed[j] = true;
                        }
                    }
                }
            }
            return nmsDetections;
        }

        private float calculateIoU(RectF box1, RectF box2) {
            float x1 = Math.max(box1.left, box2.left);
            float y1 = Math.max(box1.top, box2.top);
            float x2 = Math.min(box1.right, box2.right);
            float y2 = Math.min(box1.bottom, box2.bottom);

            float intersectionArea = Math.max(0f, x2 - x1) * Math.max(0f, y2 - y1);
            float box1Area = box1.width() * box1.height();
            float box2Area = box2.width() * box2.height();

            return intersectionArea / (box1Area + box2Area - intersectionArea);
        }

        public void close() {
            if (interpreter != null) {
                interpreter.close();
                interpreter = null;
            }
        }
    } // End of TfLiteObjectDetectionHelper class
} // End of YourService class