package com.example.temiforward;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.util.Log;
import android.view.View;

import com.robotemi.sdk.NlpResult;
import com.robotemi.sdk.Robot;
import com.robotemi.sdk.TtsRequest;
import com.robotemi.sdk.activitystream.ActivityStreamPublishMessage;
import com.robotemi.sdk.listeners.OnBeWithMeStatusChangedListener;
import com.robotemi.sdk.listeners.OnGoToLocationStatusChangedListener;
import com.robotemi.sdk.listeners.OnLocationsUpdatedListener;
import com.robotemi.sdk.listeners.OnRobotReadyListener;

import java.util.List;

public class MainActivity extends AppCompatActivity implements
        Robot.NlpListener,
        OnRobotReadyListener,
        Robot.ConversationViewAttachesListener,
        Robot.WakeupWordListener,
        Robot.ActivityStreamPublishListener,
        Robot.TtsListener,
        OnBeWithMeStatusChangedListener,
        OnGoToLocationStatusChangedListener,
        OnLocationsUpdatedListener {
    private Robot robot;
    protected void onStart() {
        super.onStart();
        Robot.getInstance().addOnRobotReadyListener(this);
        Robot.getInstance().addNlpListener(this);
        Robot.getInstance().addOnBeWithMeStatusChangedListener(this);
        Robot.getInstance().addOnGoToLocationStatusChangedListener(this);
        Robot.getInstance().addConversationViewAttachesListenerListener(this);
        Robot.getInstance().addWakeupWordListener(this);
        Robot.getInstance().addTtsListener(this);
        Robot.getInstance().addOnLocationsUpdatedListener(this);
    }

    protected void onStop() {
        super.onStop();
        Robot.getInstance().removeOnRobotReadyListener(this);
        Robot.getInstance().removeNlpListener(this);
        Robot.getInstance().removeOnBeWithMeStatusChangedListener(this);
        Robot.getInstance().removeOnGoToLocationStatusChangedListener(this);
        Robot.getInstance().removeConversationViewAttachesListenerListener(this);
        Robot.getInstance().removeWakeupWordListener(this);
        Robot.getInstance().removeTtsListener(this);
        Robot.getInstance().removeOnLocationsUpdateListener(this);
    }
    public void skidJoy(View view) {
        long t = System.currentTimeMillis();
        long end = t + 1000;
        Log.i("xyz","The function is running");
        while (System.currentTimeMillis() < end) {
            robot.skidJoy(1F, 0F);
        }



}

    @Override
    public void onPublish(ActivityStreamPublishMessage activityStreamPublishMessage) {

    }

    @Override
    public void onConversationAttaches(boolean b) {

    }

    @Override
    public void onNlpCompleted(NlpResult nlpResult) {

    }

    @Override
    public void onTtsStatusChanged(TtsRequest ttsRequest) {

    }

    @Override
    public void onWakeupWord(String s, int i) {

    }

    @Override
    public void onBeWithMeStatusChanged(String s) {

    }

    @Override
    public void onGoToLocationStatusChanged(String s, String s1, int i, String s2) {

    }

    @Override
    public void onLocationsUpdated(List<String> list) {

    }

    @Override
    public void onRobotReady(boolean b) {

    }
}