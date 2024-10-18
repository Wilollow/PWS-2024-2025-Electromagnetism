Shader "Unlit/AA"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
        // No culling or depth
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            sampler2D _MainTex;
            sampler2D _PrevFrame;
			uint _Width;
            uint _Height;

            fixed4 frag (v2f i) : SV_Target
            {
				float4 col1 = tex2D(_MainTex, i.uv + float2(0,0)/float2(_Width,_Height));
				float4 col2 = tex2D(_MainTex, i.uv + float2(1,0)/float2(_Width,_Height));
				float4 col3 = tex2D(_MainTex, i.uv + float2(0,1)/float2(_Width,_Height));
				float4 col4 = tex2D(_MainTex, i.uv + float2(1,1)/float2(_Width,_Height));

                float4 col = (col1 + col2 + col3 + col4) / 4.0f;
				
				return col;
            }
            ENDCG
        }
    }
}
