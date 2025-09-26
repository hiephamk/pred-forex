import {BrowserRouter as Routers, Routes, Route} from "react-router"
import FxDashboard from '@/Pages/FxDashboardPage'
import Dashboard from "@/Pages/Dashboard"

function App() {

  return (
    <Routers>
      <Routes>
        {/* <Route path="/" element={<LoginPage/>}/> */}
        <Route path='/' element={<Dashboard/>}/>
        <Route path='/home' element={<Dashboard/>}>
          <Route index element={<FxDashboard/>}/>
          <Route path='/home/forex' element={<FxDashboard/>}/>
        </Route>
      </Routes>
    </Routers>
  )
}

export default App
