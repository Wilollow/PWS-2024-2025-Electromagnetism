using System;
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
    private ComputeBuffer shapesBuffer;
    private ComputeBuffer materialsBuffer;
    private CustomMaterial[] orderedMaterials;
    private Shape[] orderedShapes;
    private Material frameManager;
    private Material aaManager;
    private RenderTexture rayTraceRender;
    private Camera cam;
    [SerializeField]
    private double renderedFramesCount;
    private int threadGroupsX,threadGroupsY;

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
        Shader shader2 = Shader.Find("Unlit/AA");
        aaManager = new Material(shader2);

        // Create copy of prev frame
        RenderTexture weightedAccumulation = RenderTexture.GetTemporary(src.width, src.height);
        Graphics.Blit(rayTraceRender, weightedAccumulation);

        // Run the ray tracing shader and draw the result to a temp texture
        RenderTexture currentFrame = RenderTexture.GetTemporary(src.width, src.height);
        currentFrame.enableRandomWrite = true;
        Render(src,currentFrame);

        // run anti-aliasing
        // RenderTexture aaFrame = RenderTexture.GetTemporary(src.width, src.height);
        // aaFrame.enableRandomWrite = true;
        // aaManager.SetInt("_Width", src.width);
        // aaManager.SetInt("_Height", src.height);
        // Graphics.Blit(currentFrame, aaFrame, aaManager);

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
        // RenderTexture.ReleaseTemporary(aaFrame);  
        shapesBuffer.Release();
        shapesBuffer = null;
        materialsBuffer.Release();
        materialsBuffer = null;

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
        RayTracer.SetFloat("globalTime", Mathf.Abs(Time.time + UnityEngine.Random.Range(-1f,1f)));
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
        RayTracer.SetBuffer(0, "materials", materialsBuffer);
        RayTracer.SetInt("numShapes",orderedShapes.Length);

        // Debug.Log(orderedShapes);
        // Debug.Log(orderedShapes.Length);

        RayTracer.Dispatch(0,threadGroupsX,threadGroupsY,1);
    }

    private void SceneSetup(){
        if (shapesBuffer != null) {            
            shapesBuffer.Release();
            shapesBuffer = null;
        }
        if (materialsBuffer != null) {
            materialsBuffer.Release();
            materialsBuffer = null;
        }

        CustomMaterial defaultMaterial = ScriptableObject.CreateInstance<CustomMaterial>();
        (defaultMaterial.colour.r, defaultMaterial.colour.g, defaultMaterial.colour.b) = (255 / 255, 192 / 255, 203 / 255);
        defaultMaterial.dielectric = false;
        defaultMaterial.emissionStrength = 0.0f;
        defaultMaterial.roughness = 0.5f;


        List<Shape> shapes = new List<Shape>(MonoBehaviour.FindObjectsByType<Shape>(FindObjectsSortMode.None));
        List<CustomMaterial> materials = new List<CustomMaterial>();

        shapesBuffer = new ComputeBuffer(shapes.Count,shapeInfo.getSize());
        orderedShapes = shapes.ToArray();
        // orderedShapes = orderedShapes.OrderBy(x => x.transform.position.y).ToArray();

        shapeInfo[] shapesinfo = new shapeInfo[orderedShapes.Length];
        for (int i = 0; i < orderedShapes.Length; i++)
        {
            Shape shape = orderedShapes[i];
            shapeInfo newShapeInfo = new shapeInfo
            {
                position = shape.transform.position,
                scale = shape.transform.localScale,
                shapeType = (int)shape.shapeType
            };
            CustomMaterial material = shape.material ?? defaultMaterial;
            int index = Array.IndexOf<CustomMaterial>(materials.ToArray(),material);
            if (index == -1)
            {
                materials.Add(material);
                index = materials.Count - 1;
            }
            newShapeInfo.material = index;
            // newShapeInfo.color = new Vector3(shape.color.r,shape.color.g,shape.color.b);
            // newShapeInfo.roughness = shape.roughness;
            // newShapeInfo.emission = shape.LightSourceStrength;
            shapesinfo[i] = newShapeInfo;
        }

        materialsBuffer = new ComputeBuffer(materials.Count, RayTracingMaterial.getSize());
        orderedMaterials = materials.ToArray();


        RayTracingMaterial[] materialsinfo = new RayTracingMaterial[orderedMaterials.Length];
        for (int i = 0; i < orderedMaterials.Length; i++)
        {
            CustomMaterial material = orderedMaterials[i];
            // Debug.Log(orderedMaterials[i + 2]);
            RayTracingMaterial newMaterial = new RayTracingMaterial();
            newMaterial.colour = new Vector3(material.colour.r, material.colour.g, material.colour.b);
            newMaterial.emissionStrength = material.emissionStrength;
            newMaterial.roughness = material.roughness;
            newMaterial.dielectric = Convert.ToInt32(material.dielectric);
            materialsinfo[i] = newMaterial;
        }

        materialsBuffer.SetData(materialsinfo);
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
        public int shapeType;
        public int material;

        public static int getSize(){
            return sizeof (float) * 6 + sizeof (int) * 2;
        }
    }

    struct RayTracingMaterial
    {
        public Vector3 colour;
        public float emissionStrength;
        public float roughness;
        public int dielectric;

        public static int getSize(){
            return sizeof (float) * 5 + sizeof (int) * 1;
        }
    };


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
        if (materialsBuffer != null) {
            materialsBuffer.Release();
            materialsBuffer = null;
        }
    }
}
