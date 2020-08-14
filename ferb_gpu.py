import numpy as np
import ctypes
from os import environ
environ['DISPLAY'] = ':0.0'
if not environ.get( 'PYOPENGL_PLATFORM' ):
    environ['PYOPENGL_PLATFORM'] = 'egl'
import OpenGL
from OpenGL.EGL import *
from OpenGL.GLES2 import *
from OpenGL import arrays
if environ.get( 'TEST_NO_ACCELERATE' ):
    OpenGL.USE_ACCELERATE = False

#Constants to use
TERMINATE = -1
VEC_ADD_ID = 0
MAT_MUL_ID = 1
SIGMOID_ACT_ID = 2
TANH_ACT_ID = 3	

def LoadShader(stype, shaderSrc):
    shader = glCreateShader(stype)
    if(shader == 0):
        print("Error returned 0")
        if(glGetError()!=GL_NO_ERROR):
            print('GLerror')
        return 0
    glShaderSource(shader, shaderSrc)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader))
    return shader

display, context, surface = None, None, None

configAttribs =[
    EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
    EGL_BLUE_SIZE, 8,
    EGL_GREEN_SIZE, 8,
    EGL_RED_SIZE, 8,
    EGL_DEPTH_SIZE, 8,
    EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
    EGL_NONE
]

pbufferAttribs=[
    EGL_WIDTH, 
    1000,
    EGL_HEIGHT, 
    2,
    EGL_NONE
]

contextAttribs=[
    EGL_CONTEXT_CLIENT_VERSION, 2,
    EGL_NONE
]

vertexShaderCode = '''
    attribute vec4 vPosition;
    void main() {
        gl_Position = vPosition;
    }
'''

fragmentShaderCode = {
    VEC_ADD_ID:'''
        precision highp float;
        uniform float Aarr[1000];
        uniform float Barr[1000];

        vec4 EncodeRangeV4(float value, float minVal, float maxVal) {
            value        = clamp( (value-minVal) / (maxVal-minVal), 0.0, 1.0 );
            value       *= 0.99999994039;
            vec4 encode  = fract( value * vec4(1.0, 256.0, 65536.0, 16777216.0) );
            return vec4( encode.xyz - encode.yzw / 256.0, encode.w );
        }

        void main() {
            
            int my_index = int(gl_FragCoord[0]);
            float result = Aarr[my_index] + Barr[my_index];

            gl_FragColor = EncodeRangeV4(result, -512.0, 512.0) - (1.0/1300.0);
        }
    ''',

    MAT_MUL_ID:'''
        precision highp float;
        uniform float Aarr[1000];
        uniform float Barr[1000];
        
        uniform int n;
        uniform int p;

        vec4 EncodeRangeV4(float value, float minVal, float maxVal) {
            value        = clamp( (value-minVal) / (maxVal-minVal), 0.0, 1.0 );
            value       *= .99999994039;
            vec4 encode  = fract( value * vec4(1.0, 256.0, 65536.0, 16777216.0) );
            return vec4( encode.xyz - encode.yzw / 256.0, encode.w );
        }

        void main() {
            int B_iter = int(gl_FragCoord[0]);

            float result = 0.0;

            for (int i = 0; i < n; i ++) {
                result += Aarr[i] * Barr[B_iter];
                B_iter += p;
            }
            
            gl_FragColor = EncodeRangeV4(result, -1024.0, 1024.0) - (1.0/1300.0);
        }
    ''',

    SIGMOID_ACT_ID:'''
        precision highp float;
        uniform float Aarr[1000];

        void main() {
            
            int my_index = int(gl_FragCoord[0]);
            float ex = exp(Aarr[my_index]);
            float result = ex / (1.0 + ex);

            gl_FragColor = vec4(result, 0.0, 0.0, 1.0);
        }
    ''',

    TANH_ACT_ID:'''
        precision highp float;
        uniform float Aarr[1000];

        void main() {

            int my_index = int(gl_FragCoord[0]);
            float ex = exp(Aarr[my_index]);
            float emx = exp(-Aarr[my_index]);

            float result = (ex - emx) / (ex + emx);

            result = (result + 1.0) / 2.0;

            gl_FragColor = vec4(result, 0.0, 0.0, 1.0);
        }
'''
}

vertices=(
    -1.0, -1.0, 1.0,
    -1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    -1.0, -1.0, 1.0,
    1.0, 1.0, 1.0,
    1.0, -1.0, 1.0)

prog_array = {}

def gpu_init():
    global prog_array, contextAttribs, configAttribs, display, context, surface

    display = eglGetDisplay(EGL_DEFAULT_DISPLAY)
    if(display == EGL_NO_DISPLAY):
        print("Failed to get EGL display! Error: %s", eglGetError())
        exit()

    if (eglGetError()!=12288):
        print("EGL error")

    major,minor = ctypes.c_long(), ctypes.c_long()
    if not eglInitialize( display, major, minor):
        print("Unable to initialize")
        exit()

    if (eglGetError()!=12288):
        print("EGL error")

    configAttribs = (EGLint * len(configAttribs))(*configAttribs)
    num_configs = ctypes.c_long()
    config = (EGLConfig*1)()
    eglChooseConfig(display, configAttribs, config, 1, num_configs)
    if (eglGetError()!=12288):
        print("EGL error")

    eglBindAPI(EGL_OPENGL_ES_API)
    
    pbufferAttribs_list = pbufferAttribs[:]
    pbufferAttribs_list = (EGLint * len(pbufferAttribs_list))(*pbufferAttribs_list)
    surface = eglCreatePbufferSurface(display, config[0], pbufferAttribs_list)

    if (eglGetError()!=12288):
        print("EGL error")

    contextAttribs = (EGLint * len(contextAttribs))(*contextAttribs)
    context = eglCreateContext(display, config[0], EGL_NO_CONTEXT, contextAttribs)
    if (eglGetError()!=12288):
        print("EGL error")

    eglMakeCurrent(display, surface, surface, context)

    if (eglGetError()!=12288):
        print("EGL error")
        
    desiredWidth, desiredHeight = pbufferAttribs[1], pbufferAttribs[3]
    glViewport(0, 0, desiredWidth, desiredHeight)

    if (eglGetError()!=12288):
        print("EGL error")

    glClearColor(1.0, 1.0, 1.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)	
    if(glGetError()!=GL_NO_ERROR):
        print('GLerror')
    
    for i in [VEC_ADD_ID, SIGMOID_ACT_ID, TANH_ACT_ID, MAT_MUL_ID]:
        prog_array[i] = glCreateProgram()
        if(glGetError()!=GL_NO_ERROR):
            print('GLerror')
        
        vert = LoadShader(GL_VERTEX_SHADER, vertexShaderCode)
        frag = LoadShader(GL_FRAGMENT_SHADER, fragmentShaderCode[i])
        glAttachShader(prog_array[i], vert)
        glAttachShader(prog_array[i], frag)
        glLinkProgram(prog_array[i])
        if(glGetError()!=GL_NO_ERROR):
            print('GLerror')

    glUseProgram(prog_array[0])

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER,vbo)
    verticess = (GLfloat * 18)(-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0)
    glBufferData(GL_ARRAY_BUFFER,  ctypes.sizeof(verticess), verticess, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    if(glGetError()!=GL_NO_ERROR):
        print('GLerror')
    #dims=glGetIntegerv(GL_MAX_VIEWPORT_DIMS)

def mix(x, y, a):
    res = x * (1-a) + y*a
    return res

def DecodeRangeV4(pack, minVal, maxVal) :
    vpack = [None]*4
    vpack[0] = float(pack[0])/255.0
    vpack[1] = float(pack[1])/255.0
    vpack[2] = float(pack[2])/255.0
    vpack[3] = float(pack[3])/255.0
    value = (vpack[0]/1.0) + (vpack[1]/256.0) + ((vpack[2])/(65536.0)) + ((vpack[3])/(16777216.0))
    value *= 1.0000000596
    return mix(minVal, maxVal, value)

def parse_output_buffer_vec_add(buff, n, inputs):

    for i in range(n):
        j = i<<2
        inputs[i] = DecodeRangeV4(buff[j:j+4], -512.0, 512.0)

def vec_add_gpu(A, B, C, n):
    global display, context, surface, prog_array

    glUseProgram(prog_array[VEC_ADD_ID])
    width, height = pbufferAttribs[1], pbufferAttribs[3]
    
    vec_A = glGetUniformLocation(prog_array[VEC_ADD_ID], 'Aarr')
    vec_B = glGetUniformLocation(prog_array[VEC_ADD_ID], 'Barr')
    
    A = arrays.GLfloatArray.asArray(A)
    B = arrays.GLfloatArray.asArray(B)
    glUniform1fv(vec_A, len(A), A) 
    glUniform1fv(vec_B, len(B), B)


    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), None)
    glDrawArrays(GL_TRIANGLES, 0, 18)
    eglSwapBuffers(display, surface)

    buffer = arrays.GLcharArray.asArray(np.empty(width * height * 4, np.single))
    #print("\nBuffer before : ", buffer[:20])
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, buffer)
    buffer = buffer % 256
    #print("Buffer after: ", buffer[0:20])
    parse_output_buffer_vec_add(buffer, n, C)

def parse_output_buffer_sigmoid(buff, n, inputs):
    for i in range(n):
        j = i<<2
        inputs[i] = float(buff[j]) / 255.0

def sigmoid_gpu(A, C, n):
    global display, context, surface, prog_array

    glUseProgram(prog_array[SIGMOID_ACT_ID])
    width, height = pbufferAttribs[1], pbufferAttribs[3]

    vec_A = glGetUniformLocation(prog_array[SIGMOID_ACT_ID], 'Aarr')
    A = arrays.GLfloatArray.asArray(A)
    glUniform1fv(vec_A, len(A), A) 

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), None)
    glDrawArrays(GL_TRIANGLES, 0, 18)
    eglSwapBuffers(display, surface)


    buffer = arrays.GLcharArray.asArray(np.empty(width * height * 4, np.single))
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, buffer)
    buffer = buffer % 256
    parse_output_buffer_sigmoid(buffer, n, C)

def parse_output_buffer_tanh(buff, n, inputs):
    for i in range(n):
        j = i<<2
        inputs[i] = ((float(buff[j]) / 255.0) * 2.0) - 1.0

def tanh_gpu(A, C, n):
    global display, context, surface, prog_array

    glUseProgram(prog_array[TANH_ACT_ID])
    width, height = pbufferAttribs[1], pbufferAttribs[3]
    
    vec_A = glGetUniformLocation(prog_array[TANH_ACT_ID], 'Aarr')
    A = arrays.GLfloatArray.asArray(A)
    glUniform1fv(vec_A, len(A), A) 

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), None)
    glDrawArrays(GL_TRIANGLES, 0, 18)
    eglSwapBuffers(display, surface)


    buffer = arrays.GLcharArray.asArray(np.empty(width * height * 4, np.single))
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, buffer)
    buffer = buffer % 256
    parse_output_buffer_tanh(buffer, n, C)

def parse_output_buffer_mat_mul(buff, n, inputs):
    for i in range(n):
        j = i<<2
        inputs[i] = DecodeRangeV4(buff[j:j+4], -1024.0, 1024.0)

def mat_mul_gpu(A, B, C, m, n, p):
    global display, context, surface, prog_array

    glUseProgram(prog_array[MAT_MUL_ID])

    width, height = pbufferAttribs[1], pbufferAttribs[3]

    vec_A = glGetUniformLocation(prog_array[MAT_MUL_ID], 'Aarr')
    vec_B = glGetUniformLocation(prog_array[MAT_MUL_ID], 'Barr')

    n_loc = glGetUniformLocation(prog_array[MAT_MUL_ID], 'n')
    p_loc = glGetUniformLocation(prog_array[MAT_MUL_ID], 'p')

    A = arrays.GLfloatArray.asArray(A)
    B = arrays.GLfloatArray.asArray(B)
    for i in range(m):
        glUniform1fv(vec_A, n, A[n*i:n*(i+1)]) 
        glUniform1fv(vec_B, n*p, B)		
        glUniform1i(n_loc, n)
        glUniform1i(p_loc, p)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), None)
        glDrawArrays(GL_TRIANGLES, 0, 18)
        eglSwapBuffers(display, surface)

        buffer = arrays.GLcharArray.asArray(np.empty(width * height * 4, np.single))
        glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, buffer)
        buffer = buffer % 256
        parse_output_buffer_mat_mul(buffer, p, C[p*i:p*(i+1)])
