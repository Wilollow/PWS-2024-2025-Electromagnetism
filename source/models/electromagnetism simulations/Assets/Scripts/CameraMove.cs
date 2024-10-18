using System;
using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.InputSystem;

public class CameraMove : MonoBehaviour
{
    public Transform player;
    public RayTraceRenderer rayTraceRenderer;
    public float cameraSpeed;
    public float preferredDistanceToObject, minimumDistanceToObject, zoomSensitivity;
    private float distanceToObject;
    private float zoom;
    private Vector2 mouse;
    public float degreesX;
    public float degreesY;
    private Vector3 playerOffset;
    [SerializeField]
    private float cameraLag;
    private PlayerInputs playerInputs;
    // public float turnX;
    // public float turnY;
    // Start is called before the first frame update
    void Start()
    {
        playerInputs = new PlayerInputs();
        playerInputs.CameraMovement.Enable();
        playerOffset = new Vector3(player.transform.position.x,player.transform.position.y,player.transform.position.z - preferredDistanceToObject);
    }

    // Update is called once per frame

    void Update(){
        // if ( Camera.current.name == "SceneCamera" )
        // {
        //     playerInputs = new PlayerInputs();
        //     playerInputs.CameraMovement.Enable();
        //     playerOffset = new Vector3(player.transform.position.x,player.transform.position.y,player.transform.position.z - preferredDistanceToObject);
        // }
        zoomCamera();
        getInputs();
        moveAroundPlayer();
        lookAtPlayer();
        moveToDirectSight();
    }

    private void zoomCamera(){
        float zoomInput = playerInputs.CameraMovement.Scroll.ReadValue<float>();
        if (zoomInput > 0)
        {
            zoomInput = -1;
        }else if (zoomInput < 0)
        {
            zoomInput = 1;
        }            
        

        zoom += zoomInput * zoomSensitivity;
        zoom = Mathf.Clamp(zoom, 3, 30);

        float delta = Mathf.Lerp(preferredDistanceToObject, zoom, Time.deltaTime * 10) - preferredDistanceToObject;
        if (Math.Abs(delta) < float.Epsilon * 1E41){
            delta = 0;
            preferredDistanceToObject = zoom;
            return;
        }
        preferredDistanceToObject += delta;
        rayTraceRenderer.resetFrameCount();
        // preferredDistanceToObject = Mathf.Lerp(preferredDistanceToObject, zoom, Time.deltaTime * 10);
    }

    private void moveToDirectSight(){
        int layer1 = 5;
        int layer2 = 6;
        int layerMask = (1 << layer1) | (1 << layer2);

        Vector3 cameraBottomLocal = new Vector3 (0,-1f,0);
        Vector3 adjustedCameraForward = player.position - (Camera.main.transform.position + cameraBottomLocal);

        RaycastHit hitCamera;
        bool hitObject = Physics.Raycast(player.position, -adjustedCameraForward, out hitCamera, preferredDistanceToObject + 2);

        if (!hitObject) return;
        float adjustForMinimumDistance = 0;
        distanceToObject = hitCamera.distance;
        if (distanceToObject < minimumDistanceToObject)
        {
            adjustForMinimumDistance = minimumDistanceToObject - distanceToObject;
        }
        Vector3 newCameraLocation = hitCamera.point + Camera.main.transform.forward * 2 - Camera.main.transform.forward * adjustForMinimumDistance - cameraBottomLocal;

        transform.position = newCameraLocation;
    }

    private void moveAroundPlayer(){
            float distanceX;
            float adjustDistanceX;
            float radiansX;
            float distanceY;
            float adjustDistanceY;
            float radiansY;
            degreesX += mouse.x * cameraSpeed;
            degreesX = degreesX % 360;
            radiansX =  Mathf.DeltaAngle(0,degreesX) * Mathf.Deg2Rad;
            distanceX = Mathf.Sin(radiansX/2) * 2 * preferredDistanceToObject;
            adjustDistanceX = Mathf.Acos((distanceX/2)/preferredDistanceToObject);

            degreesY += mouse.y * cameraSpeed;
            degreesY = Mathf.Clamp(degreesY,-70,10);
            radiansY =  Mathf.DeltaAngle(0,degreesY) * Mathf.Deg2Rad;
            distanceY = Mathf.Sin(radiansY/2) * 2 * preferredDistanceToObject;
            adjustDistanceY = Mathf.Acos((distanceY/2)/preferredDistanceToObject);

            Vector3 changePos = Quaternion.Euler(0,adjustDistanceX * Mathf.Rad2Deg,0) * new Vector3(0,0,distanceX);
            changePos = changePos + (Quaternion.Euler(adjustDistanceY * Mathf.Rad2Deg,-degreesX,0) * new Vector3(0,0,distanceY));

            Vector3 playerOffsetGoal = new Vector3(player.transform.position.x,player.transform.position.y,player.transform.position.z - preferredDistanceToObject);
            playerOffset = Vector3.Lerp(playerOffset,playerOffsetGoal,Time.deltaTime*cameraLag);

            transform.position = playerOffsetGoal + changePos;
            // 90 - totalAdjusted  and localposition = changepos to circle around itself

    }

    private void getInputs(){
        mouse = Mouse.current.delta.ReadValue() * .1f;
        mouse.x *= -1; 

        if (mouse != Vector2.zero)
        {
            rayTraceRenderer.resetFrameCount();
        }
    }

    private void lookAtPlayer(){
        Quaternion lookDir;
        float lookX;
        float lookZ;

        Vector3 difference = player.transform.position - transform.position;
        float differenceZ = Mathf.Sqrt(difference.x * difference.x + difference.z * difference.z);
        lookX = Mathf.Atan(difference.x/difference.z) * Mathf.Rad2Deg;
        lookZ = Mathf.Atan(difference.y/differenceZ) * Mathf.Rad2Deg;

        if (difference.z < 0)
        {
            lookX += 180;
        }

        lookDir = Quaternion.Euler(-lookZ,lookX,0);
        transform.rotation = lookDir;
    }

    public void exitFirstPerson(){
        degreesX = -player.rotation.eulerAngles.y;
        playerOffset = new Vector3(player.transform.position.x,player.transform.position.y,player.transform.position.z - preferredDistanceToObject);
    }
}
