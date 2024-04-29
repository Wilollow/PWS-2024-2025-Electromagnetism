using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PassSettings: MonoBehaviour{
    // public RenderPassEvent renderPassEvent = RenderPassEvent.AfterRendering;
    public int downsample = 1;
    public int spp = 10;
    public int maxBounces = 10;
    public int shadowResolution = 50;
    public float shadowBrightnessMul = .3f;
    public float brightness = 10;
    public float Maxbrightness = 1;
    public float minBrightness = .1f;
    public float lightSize = 0.075f;
    public float skyboxStrength,skyboxGamma;
}
