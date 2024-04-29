using System.Collections;
using System.Collections.Generic;
using UnityEngine;


[ExecuteAlways,ImageEffectAllowedInSceneView]
public class RayTraceRenderer : MonoBehaviour
{
    // public PassSettings passSettings;
    public PassSettings passSettings;
    public Texture HDRI;
    public bool activateInScene;
    private ComputeShader RayTracer;
    private Material frameManager;
    private ComputeBuffer shapesBuffer;
    private Camera cam;
    private RenderTexture rayTraceRender;
    private Shape[] orderedShapes;
    private int threadGroupsX,threadGroupsY;
    [SerializeField]
    private double renderedFramesCount;

    // Start is called before the first frame update
    void Start()
    {
        cam = Camera.main;
        RayTracer = Resources.Load<ComputeShader>("RayTracer");
        threadGroupsX = Mathf.CeilToInt(Camera.main.pixelWidth / 8.0f);
        threadGroupsY = Mathf.CeilToInt(Camera.main.pixelHeight / 8.0f);
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void OnRenderImage(RenderTexture src, RenderTexture dest)
    {
        bool isSceneCam = Camera.current.name == "SceneCamera";
        if (passSettings == null)
        {
            passSettings = new PassSettings();
        }
        rayTraceRender = initRenderTexture(rayTraceRender);

		if (isSceneCam)
		{
			if (activateInScene)
			{
                cam = Camera.current;
                Render(src,rayTraceRender);
                Graphics.Blit(rayTraceRender, dest);
                return;
			}
			else
			{
				Graphics.Blit(src, dest); // Draw the unaltered camera render to the screen
                return;
			}
		}
        Shader shader = Shader.Find("Unlit/FrameManager");
        frameManager = new Material(shader);

        // Create copy of prev frame
        RenderTexture weightedAccumulation = RenderTexture.GetTemporary(src.width, src.height);
        Graphics.Blit(rayTraceRender, weightedAccumulation);

        // Run the ray tracing shader and draw the result to a temp texture
        RenderTexture currentFrame = RenderTexture.GetTemporary(src.width, src.height);
        currentFrame.enableRandomWrite = true;
        Render(src,currentFrame);

        // Accumulate
        frameManager.SetFloat("_Frame", (float)renderedFramesCount);
        frameManager.SetTexture("_PrevFrame", weightedAccumulation);
        Graphics.Blit(currentFrame, rayTraceRender, frameManager);

        // Draw result to screen
        Graphics.Blit(rayTraceRender,dest);

        // Release temps
        RenderTexture.ReleaseTemporary(currentFrame);
        RenderTexture.ReleaseTemporary(weightedAccumulation);
        RenderTexture.ReleaseTemporary(currentFrame);      
        shapesBuffer.Release();
        shapesBuffer = null;

        renderedFramesCount += Application.isPlaying ? 1 : 0;
        
    }


    private void Render(RenderTexture src, RenderTexture destination){
        SceneSetup();

        RayTracer.SetInt("cameraWidth",cam.pixelWidth);
        RayTracer.SetInt("cameraHeight",cam.pixelHeight);
        RayTracer.SetInt("shadowResolution", passSettings.shadowResolution);
        RayTracer.SetInt("maxRecursionDepth", passSettings.maxBounces);
        RayTracer.SetInt("spp", passSettings.spp);
        RayTracer.SetFloat("shadowBrightnessMul",passSettings.shadowBrightnessMul);
        RayTracer.SetFloat("brightness", passSettings.brightness);
        RayTracer.SetFloat("Maxbrightness", passSettings.Maxbrightness);
        RayTracer.SetFloat("minBrightness", passSettings.minBrightness);
        RayTracer.SetFloat("lightRadius", passSettings.lightSize);
        RayTracer.SetFloat("globalTime", Mathf.Abs(Time.time + Random.Range(-1f,1f)));
        RayTracer.SetFloat("skyboxStrength", passSettings.skyboxStrength);
        RayTracer.SetFloat("skyboxGamma", passSettings.skyboxGamma);
        RayTracer.SetFloat("HDRIWidth", HDRI.width);
        RayTracer.SetFloat("HDRIHeight", HDRI.height);
        RayTracer.SetVector("camPos",cam.transform.position);
        RayTracer.SetVector("lightPos", GameObject.FindFirstObjectByType<Light>().transform.position);
        RayTracer.SetVector("lightCol",GameObject.FindFirstObjectByType<Light>().color);
        RayTracer.SetMatrix("_CameraInverseProjection", cam.projectionMatrix.inverse);
        RayTracer.SetMatrix("_CameraToWorld", cam.cameraToWorldMatrix);
        RayTracer.SetTexture(0,"Source",src);
        RayTracer.SetTexture(0,"Destination",destination);
        RayTracer.SetTexture(0,"skyboxTexture",HDRI);
        // RayTracer.SetTexture(0,"depthBuffer",depthBuffer);
        RayTracer.SetBuffer(0, "shapes", shapesBuffer);
        RayTracer.SetInt("numShapes",orderedShapes.Length);

        // Debug.Log(orderedShapes);
        // Debug.Log(orderedShapes.Length);

        RayTracer.Dispatch(0,threadGroupsX,threadGroupsY,1);
    }

    private void SceneSetup(){
        List<Shape> shapes = new List<Shape>(MonoBehaviour.FindObjectsByType<Shape>(FindObjectsSortMode.None));
        if (shapesBuffer != null) {            
            shapesBuffer.Release();
            shapesBuffer = null;
        }
        shapesBuffer = new ComputeBuffer(shapes.Count,shapeInfo.getSize());
        orderedShapes = shapes.ToArray();
        // orderedShapes = orderedShapes.OrderBy(x => x.transform.position.y).ToArray();

        shapeInfo[] shapesinfo = new shapeInfo[orderedShapes.Length];
        for (int i = 0; i < orderedShapes.Length; i++)
        {
            Shape shape = orderedShapes[i];
            shapeInfo newShapeInfo = new shapeInfo();
            newShapeInfo.position = shape.transform.position;
            newShapeInfo.scale = shape.transform.localScale;
            newShapeInfo.color = new Vector3(shape.color.r,shape.color.g,shape.color.b);
            newShapeInfo.shapeType = (int)shape.shapeType;
            newShapeInfo.roughness = shape.roughness;
            newShapeInfo.emission = shape.LightSourceStrength;
            shapesinfo[i] = newShapeInfo;
        }
        shapesBuffer.SetData(shapesinfo);
    }

    private RenderTexture initRenderTexture (RenderTexture texture) {
        if (texture == null || texture.width != cam.pixelWidth || texture.height != cam.pixelHeight) {
            if (texture != null) {
                texture.Release ();
            }
            resetFrameCount();
            texture = new RenderTexture (cam.pixelWidth, cam.pixelHeight, 0, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
            texture.enableRandomWrite = true;
            texture.Create ();
        }
        return texture;
    }

    struct shapeInfo{
        public Vector3 position;
        public Vector3 scale;
        public Vector3 color;
        public float emission;
        public float roughness;
        public int shapeType;

        public static int getSize(){
            return sizeof (float) * 11 + sizeof (int) * 1;
        }
    }


    public void resetFrameCount(){
        renderedFramesCount = 0;
    }

    void OnDisable()
    {
        if (shapesBuffer != null) {            
            shapesBuffer.Release();
            shapesBuffer = null;
        }
        if (rayTraceRender != null){
            rayTraceRender.Release();
            rayTraceRender = null;
        }
    }
}
