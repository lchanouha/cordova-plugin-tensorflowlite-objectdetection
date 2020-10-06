package io.wq.tensorflow;
import org.tensorflow.lite.Interpreter;
import org.apache.cordova.CordovaPlugin;
import org.apache.cordova.CallbackContext;
import org.json.JSONArray;
import org.json.JSONException;
import java.nio.MappedByteBuffer;
import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.graphics.RectF;
import android.Manifest;
import android.util.Log;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import android.util.Base64;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.BitmapFactory;
import android.media.ThumbnailUtils;
import java.util.HashMap;
import org.json.JSONArray;
import org.json.JSONObject;
import org.json.JSONException;
import io.wq.tensorflow.Classifier.Recognition;
import org.apache.cordova.PermissionHelper;
import android.content.pm.PackageManager;

public class CDVTensorFlowLite extends CordovaPlugin {
    public static final int REQUEST_CODE = 0x0ba7f;
    private Map<String,Classifier> classifiers = new HashMap();
    private String [] permissions = { Manifest.permission.CAMERA };
    private CallbackContext cc;
    @Override
    public boolean execute(String action, JSONArray args, CallbackContext callbackContext) throws JSONException {

        if (action.equals("loadModel")) {
            String modelId = args.getString(0);
            String modelFileName = args.getString(1);
         //   String assetsFileName = args.getString(2);

            if(classifiers.containsKey(modelId)){
                callbackContext.success();
                return true;
            }

            try {
                classifiers.put(modelId, TFLiteObjectDetectionAPIModel.create(
                    modelFileName.replace("file:",""),
                    300,
                    true
                ));
            } catch(IOException ioe){
                callbackContext.error(ioe.getMessage());
            }

            callbackContext.success();
            return true;
        } else if (action.equals("classify")) {
            String modelId = args.getString(0);
            String image = args.getString(1);
            byte[] imageData = Base64.decode(image, Base64.DEFAULT);
            Classifier classifier = classifiers.get(modelId);
            int size = 300;
            Bitmap bitmap = BitmapFactory.decodeByteArray(imageData, 0, imageData.length);
            Bitmap cropped = ThumbnailUtils.extractThumbnail(bitmap, size, size);
            List<Recognition> results = classifier.recognizeImage(cropped);
            JSONArray output = new JSONArray();
            try {
                for (Recognition result : results) {
                    JSONObject record = new JSONObject();
                    record.put("title", result.getTitle());
                    record.put("confidence", result.getConfidence());
                    RectF r = result.getLocation();
                    JSONArray boxes = new JSONArray();
                    boxes.put(r.top);
                    boxes.put(r.left);
                    boxes.put(r.bottom);
                    boxes.put(r.right);
                    record.put("boxes", boxes);
                    output.put(record);
                }
                callbackContext.success(output);
            } catch (JSONException e) {
                callbackContext.error(e.getMessage());
            }
        } else if (action.equals("checkPermission")){
            this.cc = callbackContext;
            if(!hasPermisssion()) {
                PermissionHelper.requestPermissions(this, 0, permissions);
            } else {
                callbackContext.success();
            }
            return true;
        }
        return false;
    }


    /**
    * processes the result of permission request
    *
    * @param requestCode The code to get request action
    * @param permissions The collection of permissions
    * @param grantResults The result of grant
    */
   public void onRequestPermissionResult(int requestCode, String[] permissions,
                                          int[] grantResults) throws JSONException
    {
        Log.i("CDVTensorFlowLote", "onRequestPermissionResult");
        for (int r : grantResults) {
            if (r == PackageManager.PERMISSION_DENIED) {
                this.cc.error("Non autorisé");
                return;
            }
        }
        this.cc.success();

    }


    /**
     * check application's permissions
     */
   public boolean hasPermisssion() {
        for(String p : permissions)
        {
            if(!PermissionHelper.hasPermission(this, p))
            {
                return false;
            }
        }
        return true;
    }
}