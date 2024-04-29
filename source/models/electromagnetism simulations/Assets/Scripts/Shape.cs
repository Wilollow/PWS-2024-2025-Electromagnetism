using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Shape : MonoBehaviour{
    public enum BehaviourType{
        add,
        subtract,
        smooth,
        mask
    }
    public enum ShapeType{
        cube,
        sphere,
    }

    public ShapeType shapeType;
    public Color color;
    public float LightSourceStrength;
    public float roughness;
    public Vector3 position{get => transform.position;}
    public Vector3 scale{get => transform.localScale;}

    private void OnDrawGizmosSelected()
    {
        if (shapeType == 0)
        {
            Gizmos.DrawWireCube(position,scale);
        }else{
            Gizmos.DrawWireSphere(position,scale.x);
            // Gizmos.DrawSphere(position,scale.x);
        }
    }
}