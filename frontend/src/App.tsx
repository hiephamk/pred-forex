import {BrowserRouter as Routers, Routes, Route} from "react-router"
import FxDashboard from '@/components/Forex/FxDashboard'

import Dashboard from '@/Pages/Dashboard'
import ExaminingHealth from '@/components/Healthcare/ExaminingHealth'
import HomePage from '@/Pages/HomePage'
import LanguagePage from '@/components/LearningLanguagues/LanguagePage'

function App() {

  return (
    <Routers>
      <Routes>
        <Route path='/' element={<Dashboard/>}/>
        <Route path='/home' element={<Dashboard/>}>
          <Route index element={<HomePage/>}/>
          <Route path='/home/forex' element={<FxDashboard/>}/>
          <Route path='/home/healthcare' element={<ExaminingHealth/>}/>
          <Route path='/home/languages' element={<LanguagePage/>}/>
        </Route>
      </Routes>
    </Routers>
  )
}

export default App
