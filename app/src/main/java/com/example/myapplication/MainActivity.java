package com.example.myapplication;

import org.tensorflow.lite.support.image.TensorImage;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.myapplication.ml.Yourmodel;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;

public class MainActivity extends AppCompatActivity {

    Button selectBtn, predictBtn, captureBtn;
    TextView result;
    ImageView imageView;
    Bitmap bitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        getPermission();

        selectBtn = findViewById(R.id.selectBtn);
        predictBtn = findViewById(R.id.predictBtn);
        captureBtn = findViewById(R.id.captureBtn);
        result = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);

        selectBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent();
                intent.setAction(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent, 10);
            }
        });

        captureBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(intent, 12);
            }
        });

//        predictBtn.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View view) {
//                try {
//                    Yourmodel model = Yourmodel.newInstance(MainActivity.this);
//                    // Creates inputs for reference.
//                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
//
//                    bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);
//
//                    inputFeature0.loadBuffer(TensorImage.fromBitmap(bitmap).getBuffer());
//
//                    // Runs model inference and gets result.
//                    Yourmodel.Outputs outputs = model.process(inputFeature0);
//
//                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
//
//                    result.setText(getMax(outputFeature0.getFloatArray())+" ");
//
//                    // Releases model resources if no longer used.
//                    model.close();
//                } catch (IOException e) {
//                    // TODO Handle the exception
//                }
//            }
//        }
            predictBtn.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    try {
                        Yourmodel model = Yourmodel.newInstance(MainActivity.this);
                        // Creates inputs for reference.
                        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);

                        bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);

//                        inputFeature0.loadBuffer(TensorImage.fromBitmap(bitmap).getBuffer());
//
//                        // Runs model inference and gets result.
//                        Yourmodel.Outputs outputs = model.process(inputFeature0);
//
//                        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
                        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
                        tensorImage.load(bitmap);
                        float[] output = model.process(tensorImage.getTensorBuffer()).getOutputFeature0AsTensorBuffer().getFloatArray();
                        result.setText(getName(getMax(output))+" ");
                        System.out.println(output);

                        // Releases model resources if no longer used.
                        model.close();
                    } catch (IOException e) {
                        // TODO Handle the exception
                    }
                }
            });

    }

    int getMax(float[] arr){
        int max = 0;
        for(int i =0; i<arr.length; i++){
            if(arr[i] > arr[max]){
                System.out.println(arr[i]);
                max = i;
            }
        }
        return max;
    }
    String getName(int index){
        if(index == 0){
            return "Cabbage";
        }if(index == 1){
            return "Capsicum";
        }if(index == 2){
            return "Tomato";
        }if(index == 3){
            return "Potato";
        }else{
            return "Carrot";
        }
    }
    void getPermission() {
        if (checkSelfPermission(android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{android.Manifest.permission.CAMERA}, 11);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == 11) {
            if (grantResults.length > 0) {
                if (grantResults[0] != PackageManager.PERMISSION_GRANTED) {
                    this.getPermission();
                }
            }
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode == 10) {
            if (data != null) {
                Uri uri = data.getData();
                try {
                    bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
                    imageView.setImageBitmap(bitmap);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }else if(requestCode == 12){
            bitmap = (Bitmap) data.getExtras().get("data");
            imageView.setImageBitmap(bitmap);
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}