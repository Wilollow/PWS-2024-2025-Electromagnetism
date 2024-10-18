using UnityEngine;

[CreateAssetMenu(fileName = "Material", menuName = "RayTracer/RayTracedMaterial", order = 1)]
public class CustomMaterial : ScriptableObject
{
    public Color colour;
    public float emissionStrength;
    public float roughness;
    public bool dielectric;
}