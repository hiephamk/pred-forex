import { useState } from 'react'
import { Box } from '@chakra-ui/react'
import './App.css'
import {BrowserRouter as Routers, Routes, Route} from "react-router"

import Dashboard from './Pages/Dashboard'

function App() {

  return (
    <Routers>
      <Routes>
        <Route path='/' element={<Dashboard/>}/>
      </Routes>

    </Routers>
  )
}

export default App
