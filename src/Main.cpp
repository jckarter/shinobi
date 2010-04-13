#include "StdAfx.hpp"

#include "Application/WindowlessApplication.hpp"
#include "Application/SDLGLApplication.hpp"

int main (int argc, char* argv[])
{

#ifdef NOWINDOW
    
    WindowlessApplication app;
    app.generateImage(argc, argv);
    SYSTEM_PAUSE

#else

    SDLGLApplication app;
    app.init(argc, argv);

    while(!app.dead())
    {
        app.fetchEvents();
        app.displayFrame();
    }

#endif


    return 0;
}
